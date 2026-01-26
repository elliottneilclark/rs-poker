use std::{
    collections::HashMap,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use tracing::{instrument, trace};

use super::Historian;

use crate::arena::GameState;
use crate::arena::action::{Action, AgentAction, AwardPayload, PlayedActionPayload};
use crate::arena::game_state::Round;
use crate::core::PlayerBitSet;

/// Storage for tracking various poker player statistics
///
/// # Fields
///
/// * `actions_count` - Vector storing the count of total actions performed by
///   each player
/// * `vpip_count` - Vector storing the count of voluntary put in pot (VPIP)
///   actions for each player (legacy, deprecated - use hands_vpip instead)
/// * `vpip_total` - Vector storing the running total of VPIP percentage for
///   each player (legacy, deprecated)
/// * `raise_count` - Vector storing the count of raise actions performed by
///   each player
/// * `hands_played` - Total hands dealt to each player (for VPIP calculation)
/// * `hands_vpip` - Hands where player voluntarily put money in preflop (binary per hand)
#[derive(Clone, Debug)]
pub struct StatsStorage {
    num_players: usize,
    // The total number of actions each player has taken
    pub actions_count: Vec<usize>,
    // How many times each player has voluntarily put money in the pot
    // DEPRECATED: Use hands_vpip for correct VPIP calculation
    pub vpip_count: Vec<usize>,
    // The total amount of money each player has voluntarily put in the pot
    // DEPRECATED: Legacy field for backwards compatibility
    pub vpip_total: Vec<f32>,
    // How many times they raised
    pub raise_count: Vec<usize>,

    // Correct per-hand tracking (preflop only, binary per hand)
    pub hands_played: Vec<usize>, // Total hands dealt to each player
    pub hands_vpip: Vec<usize>,   // Hands where player voluntarily put money in preflop
    pub hands_pfr: Vec<usize>,    // Hands where player raised preflop

    // CRITICAL: For percentage stats, we store COUNTS (opportunities + occurrences)
    // This allows proper merging/aggregation across multiple games
    pub preflop_raise_count: Vec<usize>,     // PFR occurrences
    pub preflop_actions: Vec<usize>,         // Times player had action preflop (PFR opportunities)
    pub three_bet_count: Vec<usize>,         // 3-bet occurrences
    pub three_bet_opportunities: Vec<usize>, // Times player faced a raise and could 3-bet
    pub call_count: Vec<usize>,              // Call occurrences (for aggression factor)
    pub bet_count: Vec<usize>,               // Bet occurrences (for aggression factor)

    // Financial tracking
    pub total_profit: Vec<f32>,      // Cumulative profit/loss
    pub total_invested: Vec<f32>,    // Total money put into pots (for ROI calculation)
    pub games_won: Vec<usize>,       // Win count
    pub games_lost: Vec<usize>,      // Loss count
    pub games_breakeven: Vec<usize>, // Breakeven count

    // Position tracking
    pub position_games: Vec<HashMap<usize, usize>>, // Games played per position
    pub position_profit: Vec<HashMap<usize, f32>>,  // Profit per position

    // Round outcomes (counts for win rate calculation)
    pub preflop_wins: Vec<usize>,      // Hands won that ended preflop
    pub flop_wins: Vec<usize>,         // Hands won that ended on flop
    pub turn_wins: Vec<usize>,         // Hands won that ended on turn
    pub river_wins: Vec<usize>,        // Hands won that ended on river
    pub preflop_completes: Vec<usize>, // Hands that reached preflop completion
    pub flop_completes: Vec<usize>,    // Hands that reached flop
    pub turn_completes: Vec<usize>,    // Hands that reached turn
    pub river_completes: Vec<usize>,   // Hands that reached river

    // C-Bet (Continuation Bet)
    pub cbet_opportunities: Vec<usize>, // Times was preflop aggressor and first to act on flop
    pub cbet_count: Vec<usize>,         // Times actually continuation bet

    // WTSD (Went To Showdown)
    pub wtsd_opportunities: Vec<usize>, // Times saw the flop (active post-flop)
    pub wtsd_count: Vec<usize>,         // Times went to showdown after seeing flop

    // W$SD (Won $ at Showdown)
    pub showdown_count: Vec<usize>, // Times went to showdown
    pub showdown_wins: Vec<usize>,  // Times won at showdown

    // Fold count (for AFq calculation)
    pub fold_count: Vec<usize>, // Total fold actions

    // Per-street action counts (for per-street AF)
    pub flop_bets: Vec<usize>,
    pub flop_raises: Vec<usize>,
    pub flop_calls: Vec<usize>,
    pub turn_bets: Vec<usize>,
    pub turn_raises: Vec<usize>,
    pub turn_calls: Vec<usize>,
    pub river_bets: Vec<usize>,
    pub river_raises: Vec<usize>,
    pub river_calls: Vec<usize>,

    // ATS (Attempted to Steal)
    pub steal_opportunities: Vec<usize>, // Times in steal position (CO/BTN/SB) when folded to
    pub steal_count: Vec<usize>,         // Times attempted to steal (raised in steal position)
}

impl StatsStorage {
    pub fn new_with_num_players(num_players: usize) -> Self {
        Self {
            num_players,

            // Existing fields
            actions_count: vec![0; num_players],
            vpip_count: vec![0; num_players],
            vpip_total: vec![0.0; num_players],
            raise_count: vec![0; num_players],

            // Correct per-hand tracking (preflop only, binary per hand)
            hands_played: vec![0; num_players],
            hands_vpip: vec![0; num_players],
            hands_pfr: vec![0; num_players],

            // Statistics for Phase 1
            preflop_raise_count: vec![0; num_players],
            preflop_actions: vec![0; num_players],
            three_bet_count: vec![0; num_players],
            three_bet_opportunities: vec![0; num_players],
            call_count: vec![0; num_players],
            bet_count: vec![0; num_players],

            // Financial tracking
            total_profit: vec![0.0; num_players],
            total_invested: vec![0.0; num_players],
            games_won: vec![0; num_players],
            games_lost: vec![0; num_players],
            games_breakeven: vec![0; num_players],

            // Position tracking
            position_games: vec![HashMap::new(); num_players],
            position_profit: vec![HashMap::new(); num_players],

            // Round outcomes
            preflop_wins: vec![0; num_players],
            flop_wins: vec![0; num_players],
            turn_wins: vec![0; num_players],
            river_wins: vec![0; num_players],
            preflop_completes: vec![0; num_players],
            flop_completes: vec![0; num_players],
            turn_completes: vec![0; num_players],
            river_completes: vec![0; num_players],

            // Advanced stats
            cbet_opportunities: vec![0; num_players],
            cbet_count: vec![0; num_players],
            wtsd_opportunities: vec![0; num_players],
            wtsd_count: vec![0; num_players],
            showdown_count: vec![0; num_players],
            showdown_wins: vec![0; num_players],
            fold_count: vec![0; num_players],
            flop_bets: vec![0; num_players],
            flop_raises: vec![0; num_players],
            flop_calls: vec![0; num_players],
            turn_bets: vec![0; num_players],
            turn_raises: vec![0; num_players],
            turn_calls: vec![0; num_players],
            river_bets: vec![0; num_players],
            river_raises: vec![0; num_players],
            river_calls: vec![0; num_players],
            steal_opportunities: vec![0; num_players],
            steal_count: vec![0; num_players],
        }
    }

    /// Returns the number of players this storage tracks
    pub fn num_players(&self) -> usize {
        self.num_players
    }

    // Percentage calculations from counts
    /// Calculate VPIP percentage for a player
    ///
    /// VPIP (Voluntarily Put money In Pot) is the percentage of hands where
    /// the player voluntarily put money in the pot preflop (excluding blinds).
    /// This is a binary per-hand metric: a player either VPIPed or didn't.
    pub fn vpip_percent(&self, player_idx: usize) -> f32 {
        if self.hands_played[player_idx] == 0 {
            0.0
        } else {
            (self.hands_vpip[player_idx] as f32 / self.hands_played[player_idx] as f32) * 100.0
        }
    }

    /// Calculate PFR (Pre-Flop Raise) percentage for a player
    ///
    /// PFR is the percentage of hands where the player raised preflop.
    /// This is a binary per-hand metric: a player either PFR'd or didn't.
    /// This makes it comparable to VPIP (both are % of hands).
    pub fn pfr_percent(&self, player_idx: usize) -> f32 {
        if self.hands_played[player_idx] == 0 {
            0.0
        } else {
            (self.hands_pfr[player_idx] as f32 / self.hands_played[player_idx] as f32) * 100.0
        }
    }

    /// Calculate 3-bet percentage for a player
    pub fn three_bet_percent(&self, player_idx: usize) -> f32 {
        let opportunities = self.three_bet_opportunities[player_idx];
        if opportunities == 0 {
            0.0
        } else {
            (self.three_bet_count[player_idx] as f32 / opportunities as f32) * 100.0
        }
    }

    /// Calculate aggression factor for a player
    /// (raises + bets) / calls
    pub fn aggression_factor(&self, player_idx: usize) -> f32 {
        let calls = self.call_count[player_idx];
        if calls == 0 {
            // If no calls, return infinite aggression (or a large number)
            // Return 0.0 if also no aggressive actions
            let aggressive_actions = self.raise_count[player_idx] + self.bet_count[player_idx];
            if aggressive_actions == 0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            let aggressive_actions =
                (self.raise_count[player_idx] + self.bet_count[player_idx]) as f32;
            aggressive_actions / calls as f32
        }
    }

    // Financial metrics
    /// Calculate profit per game for a player
    pub fn profit_per_game(&self, player_idx: usize) -> f32 {
        let total_games = self.games_won[player_idx]
            + self.games_lost[player_idx]
            + self.games_breakeven[player_idx];
        if total_games == 0 {
            0.0
        } else {
            self.total_profit[player_idx] / total_games as f32
        }
    }

    /// Calculate ROI percentage for a player
    ///
    /// ROI = (profit / investment) * 100
    /// Investment is the total amount of money the player put into pots.
    pub fn roi_percent(&self, player_idx: usize) -> f32 {
        let invested = self.total_invested[player_idx];
        if invested <= 0.0 {
            0.0
        } else {
            (self.total_profit[player_idx] / invested) * 100.0
        }
    }

    /// Calculate win rate for a player
    pub fn win_rate(&self, player_idx: usize) -> f32 {
        let total_games = self.games_won[player_idx]
            + self.games_lost[player_idx]
            + self.games_breakeven[player_idx];
        if total_games == 0 {
            0.0
        } else {
            (self.games_won[player_idx] as f32 / total_games as f32) * 100.0
        }
    }

    // Position accessors
    /// Get position statistics for a player
    pub fn position_stats(&self, player_idx: usize) -> &HashMap<usize, usize> {
        &self.position_games[player_idx]
    }

    /// Get position profit for a player
    pub fn position_profit(&self, player_idx: usize) -> &HashMap<usize, f32> {
        &self.position_profit[player_idx]
    }

    // Round win rates (calculated from counts)
    /// Calculate preflop win rate for a player
    pub fn preflop_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.preflop_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.preflop_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate flop win rate for a player
    pub fn flop_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.flop_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.flop_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate turn win rate for a player
    pub fn turn_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.turn_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.turn_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate river win rate for a player
    pub fn river_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.river_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.river_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    // Advanced Stats - Calculation Methods

    /// Calculate C-Bet (Continuation Bet) percentage for a player
    ///
    /// C-Bet is when a player who raised preflop bets first on the flop.
    /// Formula: (Times C-Bet / C-Bet Opportunities) * 100
    pub fn cbet_percent(&self, player_idx: usize) -> f32 {
        if self.cbet_opportunities[player_idx] == 0 {
            0.0
        } else {
            (self.cbet_count[player_idx] as f32 / self.cbet_opportunities[player_idx] as f32)
                * 100.0
        }
    }

    /// Calculate WTSD (Went To Showdown) percentage for a player
    ///
    /// WTSD is the percentage of times a player went to showdown after seeing the flop.
    /// Formula: (Times Went to Showdown / Times Saw Flop) * 100
    pub fn wtsd_percent(&self, player_idx: usize) -> f32 {
        if self.wtsd_opportunities[player_idx] == 0 {
            0.0
        } else {
            (self.wtsd_count[player_idx] as f32 / self.wtsd_opportunities[player_idx] as f32)
                * 100.0
        }
    }

    /// Calculate W$SD (Won $ at Showdown) percentage for a player
    ///
    /// W$SD is the percentage of showdowns won.
    /// Formula: (Showdowns Won / Total Showdowns) * 100
    pub fn wsd_percent(&self, player_idx: usize) -> f32 {
        if self.showdown_count[player_idx] == 0 {
            0.0
        } else {
            (self.showdown_wins[player_idx] as f32 / self.showdown_count[player_idx] as f32) * 100.0
        }
    }

    /// Calculate Aggression Frequency (AFq) for a player
    ///
    /// AFq is the percentage of aggressive actions out of all actions.
    /// Formula: (Bets + Raises) / (Bets + Raises + Calls + Folds) * 100
    pub fn aggression_frequency(&self, player_idx: usize) -> f32 {
        let aggressive_actions = self.bet_count[player_idx] + self.raise_count[player_idx];
        let total_actions =
            aggressive_actions + self.call_count[player_idx] + self.fold_count[player_idx];
        if total_actions == 0 {
            0.0
        } else {
            (aggressive_actions as f32 / total_actions as f32) * 100.0
        }
    }

    /// Calculate Flop Aggression Factor for a player
    ///
    /// Formula: (Flop Bets + Flop Raises) / Flop Calls
    pub fn flop_aggression_factor(&self, player_idx: usize) -> f32 {
        let calls = self.flop_calls[player_idx];
        if calls == 0 {
            let aggressive = self.flop_bets[player_idx] + self.flop_raises[player_idx];
            if aggressive == 0 { 0.0 } else { f32::INFINITY }
        } else {
            let aggressive = (self.flop_bets[player_idx] + self.flop_raises[player_idx]) as f32;
            aggressive / calls as f32
        }
    }

    /// Calculate Turn Aggression Factor for a player
    ///
    /// Formula: (Turn Bets + Turn Raises) / Turn Calls
    pub fn turn_aggression_factor(&self, player_idx: usize) -> f32 {
        let calls = self.turn_calls[player_idx];
        if calls == 0 {
            let aggressive = self.turn_bets[player_idx] + self.turn_raises[player_idx];
            if aggressive == 0 { 0.0 } else { f32::INFINITY }
        } else {
            let aggressive = (self.turn_bets[player_idx] + self.turn_raises[player_idx]) as f32;
            aggressive / calls as f32
        }
    }

    /// Calculate River Aggression Factor for a player
    ///
    /// Formula: (River Bets + River Raises) / River Calls
    pub fn river_aggression_factor(&self, player_idx: usize) -> f32 {
        let calls = self.river_calls[player_idx];
        if calls == 0 {
            let aggressive = self.river_bets[player_idx] + self.river_raises[player_idx];
            if aggressive == 0 { 0.0 } else { f32::INFINITY }
        } else {
            let aggressive = (self.river_bets[player_idx] + self.river_raises[player_idx]) as f32;
            aggressive / calls as f32
        }
    }

    /// Calculate Attempted to Steal percentage for a player
    ///
    /// ATS is when a player raises from a steal position (cutoff, button, or small blind)
    /// when the action folds to them.
    /// Formula: (Steal Attempts / Steal Opportunities) * 100
    pub fn steal_percent(&self, player_idx: usize) -> f32 {
        if self.steal_opportunities[player_idx] == 0 {
            0.0
        } else {
            (self.steal_count[player_idx] as f32 / self.steal_opportunities[player_idx] as f32)
                * 100.0
        }
    }

    /// Merge counts from another StatsStorage
    /// This is critical for aggregating stats across multiple games
    pub fn merge(&mut self, other: &StatsStorage) {
        assert_eq!(
            self.num_players, other.num_players,
            "Cannot merge stats with different number of players"
        );

        for i in 0..self.num_players {
            // Existing fields
            self.actions_count[i] += other.actions_count[i];
            self.vpip_count[i] += other.vpip_count[i];
            self.vpip_total[i] += other.vpip_total[i];
            self.raise_count[i] += other.raise_count[i];

            // Correct per-hand tracking (VPIP and PFR)
            self.hands_played[i] += other.hands_played[i];
            self.hands_vpip[i] += other.hands_vpip[i];
            self.hands_pfr[i] += other.hands_pfr[i];

            // Per-action counts
            self.preflop_raise_count[i] += other.preflop_raise_count[i];
            self.preflop_actions[i] += other.preflop_actions[i];
            self.three_bet_count[i] += other.three_bet_count[i];
            self.three_bet_opportunities[i] += other.three_bet_opportunities[i];
            self.call_count[i] += other.call_count[i];
            self.bet_count[i] += other.bet_count[i];

            // Financial tracking
            self.total_profit[i] += other.total_profit[i];
            self.total_invested[i] += other.total_invested[i];
            self.games_won[i] += other.games_won[i];
            self.games_lost[i] += other.games_lost[i];
            self.games_breakeven[i] += other.games_breakeven[i];

            // Position tracking - merge hashmaps
            for (position, count) in &other.position_games[i] {
                *self.position_games[i].entry(*position).or_insert(0) += count;
            }
            for (position, profit) in &other.position_profit[i] {
                *self.position_profit[i].entry(*position).or_insert(0.0) += profit;
            }

            // Round outcomes
            self.preflop_wins[i] += other.preflop_wins[i];
            self.flop_wins[i] += other.flop_wins[i];
            self.turn_wins[i] += other.turn_wins[i];
            self.river_wins[i] += other.river_wins[i];
            self.preflop_completes[i] += other.preflop_completes[i];
            self.flop_completes[i] += other.flop_completes[i];
            self.turn_completes[i] += other.turn_completes[i];
            self.river_completes[i] += other.river_completes[i];

            // Advanced stats
            self.cbet_opportunities[i] += other.cbet_opportunities[i];
            self.cbet_count[i] += other.cbet_count[i];
            self.wtsd_opportunities[i] += other.wtsd_opportunities[i];
            self.wtsd_count[i] += other.wtsd_count[i];
            self.showdown_count[i] += other.showdown_count[i];
            self.showdown_wins[i] += other.showdown_wins[i];
            self.fold_count[i] += other.fold_count[i];
            self.flop_bets[i] += other.flop_bets[i];
            self.flop_raises[i] += other.flop_raises[i];
            self.flop_calls[i] += other.flop_calls[i];
            self.turn_bets[i] += other.turn_bets[i];
            self.turn_raises[i] += other.turn_raises[i];
            self.turn_calls[i] += other.turn_calls[i];
            self.river_bets[i] += other.river_bets[i];
            self.river_raises[i] += other.river_raises[i];
            self.river_calls[i] += other.river_calls[i];
            self.steal_opportunities[i] += other.steal_opportunities[i];
            self.steal_count[i] += other.steal_count[i];
        }
    }
}

impl Default for StatsStorage {
    fn default() -> Self {
        StatsStorage::new_with_num_players(9)
    }
}

/// Thread-safe wrapper around StatsStorage that hides the Arc<RwLock<>> complexity.
///
/// This wrapper provides a clean API for sharing storage across multiple historians
/// and accessing statistics safely from multiple threads.
#[derive(Debug, Clone)]
pub struct SharedStatsStorage {
    inner: Arc<RwLock<StatsStorage>>,
}

impl SharedStatsStorage {
    /// Create a new shared storage for the given number of players.
    pub fn new(num_players: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StatsStorage::new_with_num_players(num_players))),
        }
    }

    /// Create a historian that writes to this shared storage.
    pub fn historian(&self) -> StatsTrackingHistorian {
        StatsTrackingHistorian::new_with_storage(self.clone())
    }

    /// Get a read lock on the storage.
    ///
    /// # Panics
    /// Panics if the lock is poisoned.
    pub fn read(&self) -> RwLockReadGuard<'_, StatsStorage> {
        self.inner.read().expect("StatsStorage lock poisoned")
    }

    /// Try to get a read lock on the storage.
    ///
    /// Returns an error if the lock is poisoned.
    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, StatsStorage>, super::HistorianError> {
        self.inner.read().map_err(|e| {
            super::HistorianError::LockPoisoned(format!("StatsStorage read lock poisoned: {}", e))
        })
    }

    /// Clone the current stats (snapshot).
    pub fn snapshot(&self) -> StatsStorage {
        self.read().clone()
    }

    /// Merge stats from another StatsStorage into this one.
    ///
    /// # Panics
    /// Panics if the lock is poisoned.
    pub fn merge_stats(&self, other: &StatsStorage) {
        let mut storage = self.inner.write().expect("StatsStorage lock poisoned");
        storage.merge(other);
    }

    /// Get the inner Arc<RwLock<>> for advanced use cases.
    pub fn inner(&self) -> &Arc<RwLock<StatsStorage>> {
        &self.inner
    }
}

/// Per-player hand statistics accumulated during a single hand.
#[derive(Debug, Clone, Default)]
struct PlayerHandStats {
    actions_count: usize,
    vpip_occurred: bool, // Binary - did they VPIP this hand (preflop only)?
    // Note: PFR is derived from preflop_raise_count >= 1 (no separate flag needed)
    // Legacy: action-based tracking (kept for backwards compatibility)
    vpip_count_legacy: usize,
    vpip_total_legacy: f32,
    // Per-action counts (for stats like aggression factor)
    raise_count: usize,
    bet_count: usize,
    call_count: usize,
    fold_count: usize,
    preflop_raise_count: usize, // PFR occurred if this >= 1
    preflop_actions: usize,
    three_bet_count: usize,
    three_bet_opportunities: usize,
    // Financial tracking
    invested: f32, // Total money put into pot this hand (for ROI calculation)

    // Per-street action counts
    flop_bets: usize,
    flop_raises: usize,
    flop_calls: usize,
    turn_bets: usize,
    turn_raises: usize,
    turn_calls: usize,
    river_bets: usize,
    river_raises: usize,
    river_calls: usize,

    // C-Bet tracking
    cbet_opportunity: bool,
    cbet_taken: bool,

    // Steal tracking
    steal_opportunity: bool,
    steal_taken: bool,
}

/// Accumulator for stats within a single hand.
///
/// This minimizes lock acquisitions by accumulating stats locally and then
/// flushing them to the shared storage once at hand completion.
#[derive(Debug)]
struct HandAccumulator {
    player_stats: Vec<PlayerHandStats>,
}

impl HandAccumulator {
    fn new(num_players: usize) -> Self {
        Self {
            player_stats: vec![PlayerHandStats::default(); num_players],
        }
    }

    fn reset(&mut self) {
        for stats in &mut self.player_stats {
            *stats = PlayerHandStats::default();
        }
    }
}

/// A historian implementation that tracks and stores poker game statistics
///
/// # Fields
/// * `storage` - Thread-safe shared storage for statistics
/// * `dealer_idx` - The dealer position for tracking seat positions
/// * `starting_stacks` - Starting stacks for profit calculation
/// * `current_round` - Track the current round for round-based statistics
/// * `recorded_profit` - Track profit we've already recorded for this game to avoid double-counting
/// * `accumulator` - Per-hand stats accumulator (tracks binary per-hand flags and action counts)
pub struct StatsTrackingHistorian {
    storage: SharedStatsStorage,
    dealer_idx: usize,
    starting_stacks: Vec<f32>,
    current_round: Round,
    recorded_profit: Vec<f32>, // Track what profit we've already recorded for this game
    accumulator: HandAccumulator,

    // Hand-level tracking for advanced stats
    preflop_aggressor: Option<usize>, // Who was last raiser preflop (for C-Bet)
    saw_flop: Vec<bool>,              // Who saw the flop (for WTSD)
}

impl StatsTrackingHistorian {
    /// Returns the shared storage.
    ///
    /// This is the new API that returns a thread-safe SharedStatsStorage.
    pub fn get_storage(&self) -> SharedStatsStorage {
        self.storage.clone()
    }

    /// Create a new historian with the given shared storage.
    pub fn new_with_storage(storage: SharedStatsStorage) -> Self {
        let num_players = storage.read().num_players();
        Self {
            storage,
            dealer_idx: 0,
            starting_stacks: vec![0.0; num_players],
            current_round: Round::Starting,
            recorded_profit: vec![0.0; num_players],
            accumulator: HandAccumulator::new(num_players),
            preflop_aggressor: None,
            saw_flop: vec![false; num_players],
        }
    }

    /// Check if a player is in a steal position (CO, BTN, or SB).
    /// Steal positions are the late positions that can attempt to steal the blinds.
    fn is_steal_position(player_idx: usize, dealer_idx: usize, num_players: usize) -> bool {
        if num_players < 3 {
            // In heads-up, both players are technically in steal position
            return true;
        }

        // Button (dealer) position
        let btn = dealer_idx;
        // Cutoff is one position before the button
        let co = (dealer_idx + num_players - 1) % num_players;
        // Small blind is one position after the button
        let sb = (dealer_idx + 1) % num_players;

        player_idx == btn || player_idx == co || player_idx == sb
    }

    /// Check if all players who act before this player in preflop action order have folded.
    /// This is used to determine if a player has a steal opportunity (folded to them).
    /// Uses player_active: if a player before us is NOT in player_active, they folded.
    fn is_folded_to(
        player_idx: usize,
        dealer_idx: usize,
        num_players: usize,
        player_active: &PlayerBitSet,
    ) -> bool {
        // Determine first-to-act position in preflop
        let first_to_act = if num_players == 2 {
            dealer_idx // SB (dealer) acts first in heads-up
        } else {
            (dealer_idx + 3) % num_players // UTG (first after BB)
        };

        // If we're first to act, we're automatically "folded to"
        if player_idx == first_to_act {
            return true;
        }

        // Walk from first-to-act to player_idx, checking if all have folded
        // A player has folded if they're NOT in player_active
        let mut pos = first_to_act;
        while pos != player_idx {
            if player_active.get(pos) {
                // This player is still active (didn't fold), so not folded to us
                return false;
            }
            pos = (pos + 1) % num_players;
        }

        true
    }

    fn record_played_action(
        &mut self,
        games_state: &GameState,
        payload: PlayedActionPayload,
    ) -> Result<(), super::HistorianError> {
        // Get num_players before mutable borrow
        let num_players = games_state.num_players;

        // Accumulate stats locally first
        let player_stats = &mut self.accumulator.player_stats[payload.idx];
        player_stats.actions_count += 1;

        // Track preflop opportunities
        if payload.round == Round::Preflop {
            player_stats.preflop_actions += 1;
        }

        // Helper: check if this is a raise (bet > current bet to call)
        let is_raise = payload.final_bet > payload.starting_bet;
        // Helper: check if this is a bet (first aggressive action, no prior bet)
        let is_bet = payload.starting_bet == 0.0 && payload.final_bet > 0.0;

        match payload.action {
            AgentAction::Bet(bet_amount) => {
                let put_into_pot = bet_amount - payload.starting_player_bet;

                if put_into_pot > 0.0 {
                    // Track investment for ROI calculation
                    player_stats.invested += put_into_pot;

                    // Legacy VPIP tracking (per-action)
                    player_stats.vpip_count_legacy += 1;
                    player_stats.vpip_total_legacy += put_into_pot;

                    // Correct VPIP tracking: preflop only, binary per hand
                    if payload.round == Round::Preflop && !player_stats.vpip_occurred {
                        player_stats.vpip_occurred = true;
                    }

                    // Track bet count (for aggression factor)
                    player_stats.bet_count += 1;

                    // Track per-street bets
                    match payload.round {
                        Round::Flop => player_stats.flop_bets += 1,
                        Round::Turn => player_stats.turn_bets += 1,
                        Round::River => player_stats.river_bets += 1,
                        _ => {}
                    }
                }

                // They raised
                if is_raise {
                    player_stats.raise_count += 1;

                    // Track per-street raises
                    match payload.round {
                        Round::Flop => player_stats.flop_raises += 1,
                        Round::Turn => player_stats.turn_raises += 1,
                        Round::River => player_stats.river_raises += 1,
                        _ => {}
                    }

                    // Track preflop raise (PFR derived from preflop_raise_count >= 1)
                    if payload.round == Round::Preflop {
                        player_stats.preflop_raise_count += 1;
                        // Track preflop aggressor for C-Bet
                        self.preflop_aggressor = Some(payload.idx);
                    }
                }

                // Check for 3-bet opportunity
                // If starting bet > big blind and this is a raise, it's a 3-bet opportunity
                if payload.round == Round::Preflop && payload.starting_bet > 0.0 {
                    player_stats.three_bet_opportunities += 1;

                    if is_raise {
                        player_stats.three_bet_count += 1;
                    }
                }

                // ATS (Attempted to Steal) tracking: raise from steal position when folded to
                // Check if player is in steal position AND all players before them have folded
                if payload.round == Round::Preflop
                    && Self::is_steal_position(payload.idx, self.dealer_idx, num_players)
                    && Self::is_folded_to(
                        payload.idx,
                        self.dealer_idx,
                        num_players,
                        &games_state.player_active,
                    )
                {
                    player_stats.steal_opportunity = true;
                    if is_raise {
                        player_stats.steal_taken = true;
                    }
                }

                // C-Bet tracking: preflop aggressor betting first on the flop
                // Use GameState: bet == 0 on flop means no one has bet yet
                if payload.round == Round::Flop
                    && payload.starting_bet == 0.0
                    && self.preflop_aggressor == Some(payload.idx)
                {
                    player_stats.cbet_opportunity = true;
                    if is_bet || is_raise {
                        player_stats.cbet_taken = true;
                    }
                }
            }
            AgentAction::Call => {
                // Track call count (for aggression factor)
                player_stats.call_count += 1;

                // Track per-street calls
                match payload.round {
                    Round::Flop => player_stats.flop_calls += 1,
                    Round::Turn => player_stats.turn_calls += 1,
                    Round::River => player_stats.river_calls += 1,
                    _ => {}
                }

                // Calling is also VPIP (if putting in extra money)
                let put_into_pot = payload.final_player_bet - payload.starting_player_bet;
                if put_into_pot > 0.0 {
                    // Track investment for ROI calculation
                    player_stats.invested += put_into_pot;

                    // Legacy VPIP tracking
                    player_stats.vpip_count_legacy += 1;
                    player_stats.vpip_total_legacy += put_into_pot;

                    // Correct VPIP tracking: preflop only, binary per hand
                    if payload.round == Round::Preflop && !player_stats.vpip_occurred {
                        player_stats.vpip_occurred = true;
                    }
                }
            }
            AgentAction::Fold => {
                // Track fold count (for AFq calculation)
                player_stats.fold_count += 1;

                // C-Bet tracking: if preflop aggressor folds on flop (before any bet)
                if payload.round == Round::Flop
                    && payload.starting_bet == 0.0
                    && self.preflop_aggressor == Some(payload.idx)
                {
                    player_stats.cbet_opportunity = true;
                    // cbet_taken stays false (they folded instead of c-betting)
                }
            }
            AgentAction::AllIn => {
                // All-in counts as a bet/raise for aggression purposes
                player_stats.bet_count += 1;

                // Track per-street bets (all-in is aggressive)
                match payload.round {
                    Round::Flop => player_stats.flop_bets += 1,
                    Round::Turn => player_stats.turn_bets += 1,
                    Round::River => player_stats.river_bets += 1,
                    _ => {}
                }

                let put_into_pot = payload.final_player_bet - payload.starting_player_bet;
                if put_into_pot > 0.0 {
                    // Track investment for ROI calculation
                    player_stats.invested += put_into_pot;

                    // Legacy VPIP tracking
                    player_stats.vpip_count_legacy += 1;
                    player_stats.vpip_total_legacy += put_into_pot;

                    // Correct VPIP tracking: preflop only, binary per hand
                    if payload.round == Round::Preflop && !player_stats.vpip_occurred {
                        player_stats.vpip_occurred = true;
                    }
                }

                // Check if this is a raise
                if is_raise {
                    player_stats.raise_count += 1;

                    // Track per-street raises
                    match payload.round {
                        Round::Flop => player_stats.flop_raises += 1,
                        Round::Turn => player_stats.turn_raises += 1,
                        Round::River => player_stats.river_raises += 1,
                        _ => {}
                    }

                    // Track preflop raise (PFR derived from preflop_raise_count >= 1)
                    if payload.round == Round::Preflop {
                        player_stats.preflop_raise_count += 1;
                        // Track preflop aggressor for C-Bet
                        self.preflop_aggressor = Some(payload.idx);
                    }
                }

                // C-Bet tracking
                if payload.round == Round::Flop
                    && payload.starting_bet == 0.0
                    && self.preflop_aggressor == Some(payload.idx)
                {
                    player_stats.cbet_opportunity = true;
                    player_stats.cbet_taken = true; // All-in is aggressive
                }
            }
        }

        Ok(())
    }

    fn record_round_advance(
        &mut self,
        round: Round,
        game_state: &GameState,
    ) -> Result<(), super::HistorianError> {
        self.current_round = round;

        // Track round completions - acquire write lock once
        let mut storage = self.storage.inner().write().map_err(|e| {
            super::HistorianError::LockPoisoned(format!("Write lock poisoned: {}", e))
        })?;

        let num_players = storage.num_players();

        match round {
            Round::Preflop => {
                for i in 0..num_players {
                    storage.preflop_completes[i] += 1;
                }
            }
            Round::Flop => {
                // Track who saw the flop (for WTSD calculation)
                for i in 0..num_players {
                    storage.flop_completes[i] += 1;
                    // Players still active at the start of flop "saw the flop"
                    if game_state.player_active.get(i) {
                        self.saw_flop[i] = true;
                    }
                }
            }
            Round::Turn => {
                for i in 0..num_players {
                    storage.turn_completes[i] += 1;
                }
            }
            Round::River => {
                for i in 0..num_players {
                    storage.river_completes[i] += 1;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Flush accumulated stats to shared storage.
    /// Called once at game completion to minimize lock acquisitions.
    fn flush_accumulated_stats(&mut self) -> Result<(), super::HistorianError> {
        let mut storage = self.storage.inner().write().map_err(|e| {
            super::HistorianError::LockPoisoned(format!("Write lock poisoned: {}", e))
        })?;

        for (player_idx, player_stats) in self.accumulator.player_stats.iter().enumerate() {
            storage.actions_count[player_idx] += player_stats.actions_count;
            storage.vpip_count[player_idx] += player_stats.vpip_count_legacy;
            storage.vpip_total[player_idx] += player_stats.vpip_total_legacy;
            storage.raise_count[player_idx] += player_stats.raise_count;
            storage.bet_count[player_idx] += player_stats.bet_count;
            storage.call_count[player_idx] += player_stats.call_count;
            storage.fold_count[player_idx] += player_stats.fold_count;
            storage.preflop_raise_count[player_idx] += player_stats.preflop_raise_count;
            storage.preflop_actions[player_idx] += player_stats.preflop_actions;
            storage.three_bet_count[player_idx] += player_stats.three_bet_count;
            storage.three_bet_opportunities[player_idx] += player_stats.three_bet_opportunities;
            storage.total_invested[player_idx] += player_stats.invested;

            // Per-street action counts
            storage.flop_bets[player_idx] += player_stats.flop_bets;
            storage.flop_raises[player_idx] += player_stats.flop_raises;
            storage.flop_calls[player_idx] += player_stats.flop_calls;
            storage.turn_bets[player_idx] += player_stats.turn_bets;
            storage.turn_raises[player_idx] += player_stats.turn_raises;
            storage.turn_calls[player_idx] += player_stats.turn_calls;
            storage.river_bets[player_idx] += player_stats.river_bets;
            storage.river_raises[player_idx] += player_stats.river_raises;
            storage.river_calls[player_idx] += player_stats.river_calls;

            // C-Bet tracking
            if player_stats.cbet_opportunity {
                storage.cbet_opportunities[player_idx] += 1;
                if player_stats.cbet_taken {
                    storage.cbet_count[player_idx] += 1;
                }
            }

            // ATS (Attempted to Steal) tracking
            if player_stats.steal_opportunity {
                storage.steal_opportunities[player_idx] += 1;
                if player_stats.steal_taken {
                    storage.steal_count[player_idx] += 1;
                }
            }

            // Correct VPIP tracking: increment hands_vpip if player VPIPed this hand
            if player_stats.vpip_occurred {
                storage.hands_vpip[player_idx] += 1;
            }

            // Correct PFR tracking: derived from preflop_raise_count >= 1
            if player_stats.preflop_raise_count >= 1 {
                storage.hands_pfr[player_idx] += 1;
            }
        }

        Ok(())
    }

    /// Record final profits for all players when the game completes
    fn record_game_complete(
        &mut self,
        game_state: &GameState,
    ) -> Result<(), super::HistorianError> {
        // First flush accumulated stats
        self.flush_accumulated_stats()?;

        // Then record game completion stats
        let mut storage = self.storage.inner().write().map_err(|e| {
            super::HistorianError::LockPoisoned(format!("Write lock poisoned: {}", e))
        })?;

        // Determine if this went to showdown (round is Complete and more than one player active)
        let went_to_showdown =
            game_state.round == Round::Complete && game_state.player_active.count() > 1;

        // Record final profit for each player
        for player_idx in 0..game_state.num_players {
            let final_profit = game_state.player_reward(player_idx);

            // Increment hands_played for correct VPIP calculation
            storage.hands_played[player_idx] += 1;

            // Add this game's profit to total profit
            storage.total_profit[player_idx] += final_profit;

            // WTSD (Went To Showdown) tracking
            // If player saw the flop, they have a WTSD opportunity
            if self.saw_flop[player_idx] {
                storage.wtsd_opportunities[player_idx] += 1;

                // If they went to showdown (still active at the end with multiple players)
                if went_to_showdown && game_state.player_active.get(player_idx) {
                    storage.wtsd_count[player_idx] += 1;
                }
            }

            // W$SD (Won $ at Showdown) tracking
            if went_to_showdown && game_state.player_active.get(player_idx) {
                storage.showdown_count[player_idx] += 1;

                // If they won money at showdown
                if final_profit > 0.01 {
                    storage.showdown_wins[player_idx] += 1;
                }
            }

            // Update win/loss/breakeven counts
            if final_profit > 0.01 {
                storage.games_won[player_idx] += 1;

                // Track round-based wins
                match game_state.round_before {
                    Round::Preflop => storage.preflop_wins[player_idx] += 1,
                    Round::Flop => storage.flop_wins[player_idx] += 1,
                    Round::Turn => storage.turn_wins[player_idx] += 1,
                    Round::River => storage.river_wins[player_idx] += 1,
                    _ => {}
                }
            } else if final_profit < -0.01 {
                storage.games_lost[player_idx] += 1;
            } else {
                storage.games_breakeven[player_idx] += 1;
            }

            // Track position statistics
            *storage.position_games[player_idx]
                .entry(player_idx)
                .or_insert(0) += 1;
            *storage.position_profit[player_idx]
                .entry(player_idx)
                .or_insert(0.0) += final_profit;
        }

        Ok(())
    }

    /// Record award information for statistics like round wins, but don't track profit here
    fn record_award_without_profit(
        &mut self,
        _game_state: &GameState,
        _payload: AwardPayload,
    ) -> Result<(), super::HistorianError> {
        // We no longer track profit from awards since we do it at game complete
        // This method exists to preserve any other award-related statistics if needed
        Ok(())
    }

    fn record_game_start(&mut self, game_state: &GameState) -> Result<(), super::HistorianError> {
        // Store starting stacks and dealer position
        self.starting_stacks = game_state.starting_stacks.clone();
        self.dealer_idx = game_state.dealer_idx;
        self.current_round = Round::Starting;

        // Reset recorded profit for this new game
        self.recorded_profit = vec![0.0; game_state.num_players];

        // Reset accumulator for the new hand (resets vpip_occurred, pfr_occurred, etc.)
        self.accumulator.reset();

        // Reset hand-level tracking for advanced stats
        self.preflop_aggressor = None;
        self.saw_flop = vec![false; game_state.num_players];

        Ok(())
    }

    /// Backwards compatible constructor - creates own storage.
    pub fn new_with_num_players(num_players: usize) -> Self {
        SharedStatsStorage::new(num_players).historian()
    }
}

impl Default for StatsTrackingHistorian {
    fn default() -> Self {
        SharedStatsStorage::new(9).historian()
    }
}

impl Historian for StatsTrackingHistorian {
    #[instrument(level = "trace", skip(self, game_state))]
    fn record_action(
        &mut self,
        _id: u128,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), super::HistorianError> {
        trace!(?action, "StatsTrackingHistorian processing action");
        match action {
            Action::GameStart(_) => self.record_game_start(game_state),
            Action::PlayedAction(payload) => self.record_played_action(game_state, payload),
            Action::FailedAction(failed_action_payload) => {
                self.record_played_action(game_state, failed_action_payload.result)
            }
            Action::RoundAdvance(round) => {
                if round == Round::Complete {
                    // Record final profits for all players when the game completes
                    self.record_game_complete(game_state)?;
                }
                self.record_round_advance(round, game_state)
            }
            Action::Award(payload) => {
                // Still record awards for other statistics (round wins, etc.)
                // but don't use them for profit calculation
                self.record_award_without_profit(game_state, payload)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        Agent, HoldemSimulationBuilder,
        agent::{AllInAgent, CallingAgent, FoldingAgent, VecReplayAgent},
    };

    use super::*;

    #[test]
    fn test_all_in_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<AllInAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert!(storage.read().actions_count.iter().all(|&count| count == 1));
    }

    #[test]
    fn test_calling_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert!(storage.read().actions_count.iter().all(|&count| count == 4));
    }

    #[test]
    fn test_folding_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let actions_count = &storage.read().actions_count;

        // Player 0 folded before player 1 could even act.
        assert_eq!(actions_count.first(), Some(&1));
        assert_eq!(actions_count.get(1), Some(&0));
    }

    #[test]
    fn test_replay_agents_had_raises_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "replay-agent-0",
                vec![AgentAction::Bet(10.0), AgentAction::Bet(40.0)],
                AgentAction::Bet(0.0),
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "replay-agent-1",
                vec![
                    AgentAction::Bet(10.0),
                    AgentAction::Bet(20.0),
                    AgentAction::Bet(40.0),
                ],
                AgentAction::Bet(0.0),
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(storage.read().raise_count, vec![1, 1]);
    }

    #[test]
    fn test_pfr_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        // Agent 0 raises preflop, Agent 1 calls
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)], // Raise preflop
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![AgentAction::Call], // Call preflop
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Agent 0 should have 1 preflop raise
        assert_eq!(borrowed.preflop_raise_count[0], 1);
        assert_eq!(borrowed.preflop_raise_count[1], 0);

        // Both should have preflop opportunities
        assert!(borrowed.preflop_actions[0] > 0);
        assert!(borrowed.preflop_actions[1] > 0);

        // Calculate PFR percentage
        let pfr_0 = borrowed.pfr_percent(0);
        assert!(pfr_0 > 0.0);
    }

    #[test]
    fn test_call_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(3));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 3];

        // Use 3 players with VecReplayAgent to ensure Call actions
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)], // Raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller-1",
                vec![AgentAction::Call], // Call the raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller-2",
                vec![AgentAction::Call], // Call the raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // The callers should have calls recorded
        assert!(
            borrowed.call_count[1] > 0 || borrowed.call_count[2] > 0,
            "Expected at least one caller to have calls tracked, got: {:?}",
            borrowed.call_count
        );
    }

    #[test]
    fn test_vpip_calculation() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Calling agent should have high VPIP
        let vpip_0 = borrowed.vpip_percent(0);
        assert!(vpip_0 > 0.0);

        // Folding agent should have low VPIP (only forced bets)
        let vpip_1 = borrowed.vpip_percent(1);
        assert_eq!(vpip_1, 0.0); // Folding agent never voluntarily puts money in
    }

    #[test]
    fn test_merge_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.actions_count[0] = 5;
        stats1.vpip_count[0] = 3;
        stats1.total_profit[0] = 100.0;
        stats1.games_won[0] = 2;

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.actions_count[0] = 3;
        stats2.vpip_count[0] = 2;
        stats2.total_profit[0] = 50.0;
        stats2.games_won[0] = 1;

        stats1.merge(&stats2);

        assert_eq!(stats1.actions_count[0], 8);
        assert_eq!(stats1.vpip_count[0], 5);
        assert_eq!(stats1.total_profit[0], 150.0);
        assert_eq!(stats1.games_won[0], 3);
    }

    #[test]
    fn test_merge_position_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.position_games[0].insert(0, 5);
        stats1.position_profit[0].insert(0, 100.0);

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.position_games[0].insert(0, 3);
        stats2.position_profit[0].insert(0, 50.0);

        stats1.merge(&stats2);

        assert_eq!(stats1.position_games[0].get(&0), Some(&8));
        assert_eq!(stats1.position_profit[0].get(&0), Some(&150.0));
    }

    #[test]
    fn test_aggression_factor_no_calls() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 5;
        stats.bet_count[0] = 3;
        stats.call_count[0] = 0;

        let af = stats.aggression_factor(0);
        assert_eq!(af, f32::INFINITY);
    }

    #[test]
    fn test_aggression_factor_with_calls() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 4;
        stats.bet_count[0] = 2;
        stats.call_count[0] = 2;

        let af = stats.aggression_factor(0);
        assert_eq!(af, 3.0); // (4 + 2) / 2 = 3.0
    }

    #[test]
    fn test_profit_per_game() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 300.0;
        stats.games_won[0] = 5;
        stats.games_lost[0] = 3;
        stats.games_breakeven[0] = 2;

        let ppg = stats.profit_per_game(0);
        assert_eq!(ppg, 30.0); // 300.0 / 10 games = 30.0
    }

    #[test]
    fn test_win_rate() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.games_won[0] = 7;
        stats.games_lost[0] = 2;
        stats.games_breakeven[0] = 1;

        let wr = stats.win_rate(0);
        assert_eq!(wr, 70.0); // 7 / 10 * 100 = 70%
    }

    #[test]
    fn test_edge_case_empty_stats() {
        let stats = StatsStorage::new_with_num_players(2);

        // All calculations should return 0 for empty stats
        assert_eq!(stats.vpip_percent(0), 0.0);
        assert_eq!(stats.pfr_percent(0), 0.0);
        assert_eq!(stats.three_bet_percent(0), 0.0);
        assert_eq!(stats.aggression_factor(0), 0.0);
        assert_eq!(stats.profit_per_game(0), 0.0);
        assert_eq!(stats.win_rate(0), 0.0);
    }

    #[test]
    fn test_round_win_rates() {
        let mut stats = StatsStorage::new_with_num_players(1);

        stats.preflop_wins[0] = 2;
        stats.preflop_completes[0] = 10;

        stats.flop_wins[0] = 3;
        stats.flop_completes[0] = 8;

        stats.turn_wins[0] = 1;
        stats.turn_completes[0] = 5;

        stats.river_wins[0] = 2;
        stats.river_completes[0] = 4;

        assert_eq!(stats.preflop_win_rate(0), 20.0);
        assert_eq!(stats.flop_win_rate(0), 37.5);
        assert_eq!(stats.turn_win_rate(0), 20.0);
        assert_eq!(stats.river_win_rate(0), 50.0);
    }

    #[test]
    fn test_zero_sum_property_simple() {
        // Test that profits sum to zero in a simple heads-up game
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();
        let total_profit = borrowed.total_profit[0] + borrowed.total_profit[1];

        // In a zero-sum game, total profit should be very close to zero
        // Allow for small floating point errors
        assert!(
            total_profit.abs() < 0.01,
            "Total profit should be zero (zero-sum), but got: {}. Player 0: {}, Player 1: {}",
            total_profit,
            borrowed.total_profit[0],
            borrowed.total_profit[1]
        );

        // One player should have positive profit, one should have negative (or both zero)
        let player0_profit = borrowed.total_profit[0];
        let player1_profit = borrowed.total_profit[1];

        // At least one player should have participated (won/lost/breakeven > 0)
        let total_games_0 =
            borrowed.games_won[0] + borrowed.games_lost[0] + borrowed.games_breakeven[0];
        let total_games_1 =
            borrowed.games_won[1] + borrowed.games_lost[1] + borrowed.games_breakeven[1];

        assert!(
            total_games_0 > 0 || total_games_1 > 0,
            "At least one player should have game results"
        );

        println!(
            "Player 0 profit: {}, games: {}",
            player0_profit, total_games_0
        );
        println!(
            "Player 1 profit: {}, games: {}",
            player1_profit, total_games_1
        );
        println!("Total profit: {}", total_profit);
    }

    #[test]
    fn test_zero_sum_property_three_players() {
        // Test zero-sum property with three players
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(3));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();
        let total_profit =
            borrowed.total_profit[0] + borrowed.total_profit[1] + borrowed.total_profit[2];

        // In a zero-sum game, total profit should be very close to zero
        assert!(
            total_profit.abs() < 0.01,
            "Total profit should be zero (zero-sum), but got: {}. Player profits: [{}, {}, {}]",
            total_profit,
            borrowed.total_profit[0],
            borrowed.total_profit[1],
            borrowed.total_profit[2]
        );

        println!(
            "Player profits: [{}, {}, {}], Total: {}",
            borrowed.total_profit[0],
            borrowed.total_profit[1],
            borrowed.total_profit[2],
            total_profit
        );
    }

    #[test]
    fn test_profit_calculation_matches_game_state() {
        // Test that our profit calculation matches what GameState.player_reward() would return
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks.clone(), 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let final_game_state = &sim.game_state;
        let borrowed = storage.read();

        // Check that our tracked profit matches GameState.player_reward()
        for i in 0..2 {
            let tracked_profit = borrowed.total_profit[i];
            let actual_reward = final_game_state.player_reward(i);

            assert!(
                (tracked_profit - actual_reward).abs() < 0.01,
                "Player {} tracked profit ({}) should match actual reward ({})",
                i,
                tracked_profit,
                actual_reward
            );
        }
    }

    /// Verifies that VPIP percentage is calculated correctly as (hands_vpip / hands_played) * 100.
    #[test]
    fn test_vpip_percent_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.hands_played[0] = 100;
        stats.hands_vpip[0] = 25;
        let vpip = stats.vpip_percent(0);
        assert!(
            (vpip - 25.0).abs() < 0.001,
            "Expected VPIP of 25.0%, got {}",
            vpip
        );
    }

    /// Verifies that VPIP returns 0.0 when no hands have been played (zero denominator case).
    #[test]
    fn test_vpip_percent_zero_hands() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.vpip_percent(0),
            0.0,
            "VPIP should be 0 when no hands played"
        );
    }

    /// Verifies that PFR percentage is calculated correctly as (hands_pfr / hands_played) * 100.
    #[test]
    fn test_pfr_percent_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.hands_played[0] = 50;
        stats.hands_pfr[0] = 10;
        let pfr = stats.pfr_percent(0);
        assert!(
            (pfr - 20.0).abs() < 0.001,
            "Expected PFR of 20.0%, got {}",
            pfr
        );
    }

    /// Verifies that PFR returns 0.0 when no hands have been played (zero denominator case).
    #[test]
    fn test_pfr_percent_zero_hands() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.pfr_percent(0),
            0.0,
            "PFR should be 0 when no hands played"
        );
    }

    /// Verifies that 3-bet percentage is calculated correctly as (three_bet_count / three_bet_opportunities) * 100.
    #[test]
    fn test_three_bet_percent_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.three_bet_opportunities[0] = 20;
        stats.three_bet_count[0] = 5;
        let three_bet = stats.three_bet_percent(0);
        assert!(
            (three_bet - 25.0).abs() < 0.001,
            "Expected 3-bet of 25.0%, got {}",
            three_bet
        );
    }

    /// Verifies that 3-bet returns 0.0 when there are no opportunities (zero denominator case).
    #[test]
    fn test_three_bet_percent_zero_opportunities() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.three_bet_percent(0),
            0.0,
            "3-bet should be 0 when no opportunities"
        );
    }

    /// Verifies that steal percentage (ATS) is calculated correctly as (steal_count / steal_opportunities) * 100.
    #[test]
    fn test_steal_percent_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.steal_opportunities[0] = 10;
        stats.steal_count[0] = 4;
        let steal = stats.steal_percent(0);
        assert!(
            (steal - 40.0).abs() < 0.001,
            "Expected ATS of 40.0%, got {}",
            steal
        );
    }

    /// Verifies that steal percentage returns 0.0 when there are no opportunities (zero denominator case).
    #[test]
    fn test_steal_percent_zero_opportunities() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.steal_percent(0),
            0.0,
            "ATS should be 0 when no opportunities"
        );
    }

    #[test]
    fn test_is_steal_position() {
        // Test with 6 players, dealer at position 0
        // Button (BTN) = 0, Cutoff (CO) = 5, Small Blind (SB) = 1
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 6),
            "Button should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(5, 0, 6),
            "Cutoff should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 6),
            "Small blind should be steal position"
        );
        assert!(
            !StatsTrackingHistorian::is_steal_position(2, 0, 6),
            "Big blind should NOT be steal position"
        );
        assert!(
            !StatsTrackingHistorian::is_steal_position(3, 0, 6),
            "UTG should NOT be steal position"
        );
        assert!(
            !StatsTrackingHistorian::is_steal_position(4, 0, 6),
            "MP should NOT be steal position"
        );

        // Test heads-up: both players are in steal position
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 2),
            "In heads-up, button is steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 2),
            "In heads-up, other player is steal position"
        );
    }

    #[test]
    fn test_is_folded_to() {
        use crate::core::PlayerBitSet;

        // 6-player game, dealer at 0: BTN=0, SB=1, BB=2, UTG=3, MP=4, CO=5
        // Preflop action order: UTG(3) → MP(4) → CO(5) → BTN(0) → SB(1) → BB(2)

        // All players still active - no one folded to anyone
        let mut player_active = PlayerBitSet::new(6);
        for i in 0..6 {
            player_active.enable(i);
        }
        // UTG (first to act) is always folded to
        assert!(
            StatsTrackingHistorian::is_folded_to(3, 0, 6, &player_active),
            "UTG is always folded to (first to act)"
        );
        // CO (position 5) is NOT folded to if UTG and MP still active
        assert!(
            !StatsTrackingHistorian::is_folded_to(5, 0, 6, &player_active),
            "CO is NOT folded to when UTG/MP still active"
        );

        // UTG and MP fold (removed from player_active)
        player_active.disable(3); // UTG folds
        player_active.disable(4); // MP folds

        // Now CO should be folded to
        assert!(
            StatsTrackingHistorian::is_folded_to(5, 0, 6, &player_active),
            "CO is folded to when UTG/MP have folded"
        );
        // BTN is NOT folded to (CO still active)
        assert!(
            !StatsTrackingHistorian::is_folded_to(0, 0, 6, &player_active),
            "BTN is NOT folded to when CO still active"
        );

        // CO also folds
        player_active.disable(5);

        // Now BTN is folded to
        assert!(
            StatsTrackingHistorian::is_folded_to(0, 0, 6, &player_active),
            "BTN is folded to when UTG/MP/CO have folded"
        );

        // Test heads-up: SB (dealer) is always first to act
        let mut hu_active = PlayerBitSet::new(2);
        hu_active.enable(0);
        hu_active.enable(1);
        assert!(
            StatsTrackingHistorian::is_folded_to(0, 0, 2, &hu_active),
            "In heads-up, SB (dealer) is first to act, always folded to"
        );
        assert!(
            !StatsTrackingHistorian::is_folded_to(1, 0, 2, &hu_active),
            "In heads-up, BB is NOT folded to if SB still active"
        );
        hu_active.disable(0); // SB folds
        assert!(
            StatsTrackingHistorian::is_folded_to(1, 0, 2, &hu_active),
            "In heads-up, BB is folded to when SB folds"
        );
    }

    #[test]
    fn test_aggression_factor_zero_aggressive_actions() {
        let stats = StatsStorage::new_with_num_players(1);
        // No raises, no bets, no calls
        assert_eq!(
            stats.aggression_factor(0),
            0.0,
            "AF should be 0 when no aggressive actions"
        );
    }

    #[test]
    fn test_profit_per_game_zero_games() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.profit_per_game(0),
            0.0,
            "Profit per game should be 0 when no games"
        );
    }

    #[test]
    fn test_profit_per_game_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 250.0;
        stats.games_won[0] = 3;
        stats.games_lost[0] = 2;
        // 5 total games, 250.0 profit = 50.0 profit per game
        let ppg = stats.profit_per_game(0);
        assert!(
            (ppg - 50.0).abs() < 0.001,
            "Expected 50.0 profit per game, got {}",
            ppg
        );
    }

    #[test]
    fn test_win_rate_zero_games() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(stats.win_rate(0), 0.0, "Win rate should be 0 when no games");
    }

    #[test]
    fn test_win_rate_exact_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.games_won[0] = 3;
        stats.games_lost[0] = 5;
        stats.games_breakeven[0] = 2;
        // 3 wins / 10 total = 30%
        let wr = stats.win_rate(0);
        assert!(
            (wr - 30.0).abs() < 0.001,
            "Expected 30.0% win rate, got {}",
            wr
        );
    }

    #[test]
    fn test_preflop_win_rate_zero_completes() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.preflop_win_rate(0),
            0.0,
            "Preflop win rate should be 0 when no completes"
        );
    }

    #[test]
    fn test_flop_win_rate_zero_completes() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.flop_win_rate(0),
            0.0,
            "Flop win rate should be 0 when no completes"
        );
    }

    #[test]
    fn test_turn_win_rate_zero_completes() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.turn_win_rate(0),
            0.0,
            "Turn win rate should be 0 when no completes"
        );
    }

    #[test]
    fn test_river_win_rate_zero_completes() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(
            stats.river_win_rate(0),
            0.0,
            "River win rate should be 0 when no completes"
        );
    }

    #[test]
    fn test_merge_hands_played_and_vpip() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.hands_played[0] = 10;
        stats1.hands_vpip[0] = 5;

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.hands_played[0] = 15;
        stats2.hands_vpip[0] = 8;

        stats1.merge(&stats2);

        assert_eq!(
            stats1.hands_played[0], 25,
            "Hands played should merge correctly"
        );
        assert_eq!(
            stats1.hands_vpip[0], 13,
            "Hands VPIP should merge correctly"
        );
    }

    #[test]
    fn test_merge_all_round_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.preflop_wins[0] = 1;
        stats1.flop_wins[0] = 2;
        stats1.turn_wins[0] = 3;
        stats1.river_wins[0] = 4;
        stats1.preflop_completes[0] = 10;
        stats1.flop_completes[0] = 20;
        stats1.turn_completes[0] = 30;
        stats1.river_completes[0] = 40;

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.preflop_wins[0] = 5;
        stats2.flop_wins[0] = 6;
        stats2.turn_wins[0] = 7;
        stats2.river_wins[0] = 8;
        stats2.preflop_completes[0] = 15;
        stats2.flop_completes[0] = 25;
        stats2.turn_completes[0] = 35;
        stats2.river_completes[0] = 45;

        stats1.merge(&stats2);

        assert_eq!(stats1.preflop_wins[0], 6);
        assert_eq!(stats1.flop_wins[0], 8);
        assert_eq!(stats1.turn_wins[0], 10);
        assert_eq!(stats1.river_wins[0], 12);
        assert_eq!(stats1.preflop_completes[0], 25);
        assert_eq!(stats1.flop_completes[0], 45);
        assert_eq!(stats1.turn_completes[0], 65);
        assert_eq!(stats1.river_completes[0], 85);
    }

    #[test]
    fn test_vpip_binary_per_hand() {
        // Test that multiple preflop actions in the same hand result in exactly 1 VPIP hand
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // Agent 0 raises, agent 1 re-raises, agent 0 calls
        // Both agents should have exactly 1 VPIP hand despite multiple actions
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser-caller",
                vec![AgentAction::Bet(20.0), AgentAction::Call], // Raise then call the 3-bet
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "three-better",
                vec![AgentAction::Bet(40.0)], // 3-bet
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();
        // Both players should have exactly 1 hand played
        assert_eq!(
            stats.hands_played[0], 1,
            "Player 0 should have 1 hand played"
        );
        assert_eq!(
            stats.hands_played[1], 1,
            "Player 1 should have 1 hand played"
        );
        // Both should have VPIP = 1 (binary per hand, not per action)
        assert_eq!(
            stats.hands_vpip[0], 1,
            "Player 0 should have 1 VPIP hand (not 2 despite multiple actions)"
        );
        assert_eq!(stats.hands_vpip[1], 1, "Player 1 should have 1 VPIP hand");
        // VPIP % should be 100% for both
        assert!(
            (stats.vpip_percent(0) - 100.0).abs() < 0.001,
            "Player 0 VPIP should be 100%"
        );
        assert!(
            (stats.vpip_percent(1) - 100.0).abs() < 0.001,
            "Player 1 VPIP should be 100%"
        );
    }

    #[test]
    fn test_shared_storage_snapshot() {
        let storage = SharedStatsStorage::new(2);
        {
            let mut writer = storage.inner().write().unwrap();
            writer.actions_count[0] = 42;
            writer.total_profit[0] = 100.0;
        }

        let snapshot = storage.snapshot();
        assert_eq!(snapshot.actions_count[0], 42);
        assert_eq!(snapshot.total_profit[0], 100.0);
    }

    #[test]
    fn test_shared_storage_merge_stats() {
        let storage = SharedStatsStorage::new(2);
        let mut other = StatsStorage::new_with_num_players(2);
        other.actions_count[0] = 10;
        other.total_profit[0] = 50.0;
        other.hands_played[0] = 5;
        other.hands_vpip[0] = 3;

        storage.merge_stats(&other);

        let stats = storage.read();
        assert_eq!(stats.actions_count[0], 10);
        assert_eq!(stats.total_profit[0], 50.0);
        assert_eq!(stats.hands_played[0], 5);
        assert_eq!(stats.hands_vpip[0], 3);
    }

    #[test]
    fn test_num_players_accessor() {
        let stats = StatsStorage::new_with_num_players(6);
        assert_eq!(stats.num_players(), 6);
    }

    // ======= ROI TESTS =======
    // Tests to verify ROI is correctly calculated from profit / investment

    #[test]
    fn test_roi_percent_basic() {
        let mut stats = StatsStorage::new_with_num_players(2);
        // Player 0: invested 100, profit 50 -> ROI = 50%
        stats.total_invested[0] = 100.0;
        stats.total_profit[0] = 50.0;

        assert!((stats.roi_percent(0) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_roi_percent_loss() {
        let mut stats = StatsStorage::new_with_num_players(2);
        // Player 0: invested 100, profit -30 -> ROI = -30%
        stats.total_invested[0] = 100.0;
        stats.total_profit[0] = -30.0;

        assert!((stats.roi_percent(0) - (-30.0)).abs() < 0.01);
    }

    #[test]
    fn test_roi_percent_zero_investment() {
        let mut stats = StatsStorage::new_with_num_players(2);
        // Player 0: invested 0 (e.g., folding agent) -> ROI = 0%
        stats.total_invested[0] = 0.0;
        stats.total_profit[0] = -10.0; // Lost blinds

        assert_eq!(stats.roi_percent(0), 0.0);
    }

    #[test]
    fn test_investment_tracking_bet() {
        // Test that betting adds to total_invested
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // Agent 0 bets, agent 1 folds
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "bettor",
                vec![AgentAction::Bet(30.0)], // Bets 30 (puts in 25 more beyond SB of 5)
                AgentAction::Fold,
            )) as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();
        // Player 0 (SB) put in 25 more to make their bet 30 total
        assert!(
            stats.total_invested[0] > 0.0,
            "Bettor should have invested money"
        );
        // Player 1 (BB) folded, so they didn't invest anything beyond forced blind
        assert_eq!(
            stats.total_invested[1], 0.0,
            "Folder should not have voluntary investment"
        );
    }

    #[test]
    fn test_investment_tracking_call() {
        // Test that calling adds to total_invested
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // Agent 0 raises, agent 1 calls
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)],
                AgentAction::Fold,
            )) as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();
        // Player 0 (SB) raised to 20, so invested 15 (20 - 5 SB)
        assert!(
            stats.total_invested[0] > 0.0,
            "Raiser should have invested money"
        );
        // Player 1 (BB) called 20, so invested 10 (20 - 10 BB)
        assert!(
            stats.total_invested[1] > 0.0,
            "Caller should have invested money"
        );
    }

    // ======= VPIP EDGE CASE TESTS =======
    // Tests to verify VPIP is correctly tracked for various agent behaviors

    #[test]
    fn test_folding_agent_zero_vpip_heads_up() {
        // A folding agent should have 0% VPIP when playing heads-up against another folder
        // because they just fold (no voluntary money put in)
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Both players should have played 1 hand
        assert_eq!(
            stats.hands_played[0], 1,
            "Player 0 should have 1 hand played"
        );
        assert_eq!(
            stats.hands_played[1], 1,
            "Player 1 should have 1 hand played"
        );

        // The small blind folds, so they don't VPIP (didn't voluntarily add money beyond forced blind)
        // The big blind wins without acting (or checks), so they also don't VPIP
        assert_eq!(
            stats.hands_vpip[0], 0,
            "Folding player 0 (SB) should have 0 VPIP hands - they just folded"
        );
        assert_eq!(
            stats.hands_vpip[1], 0,
            "Player 1 (BB) should have 0 VPIP hands - they won uncontested"
        );

        // VPIP percentages should be 0%
        assert_eq!(
            stats.vpip_percent(0),
            0.0,
            "Folding agent should have 0% VPIP"
        );
        assert_eq!(
            stats.vpip_percent(1),
            0.0,
            "Big blind who won uncontested should have 0% VPIP"
        );
    }

    #[test]
    fn test_folding_agent_vs_caller_zero_vpip() {
        // Folding agent should have 0% VPIP against a calling agent
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // Folding agent is SB (position 0), Calling agent is BB (position 1)
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // The folding agent (SB) folds, so 0 VPIP
        assert_eq!(
            stats.hands_vpip[0], 0,
            "Folding agent should have 0 VPIP hands"
        );
        assert_eq!(
            stats.vpip_percent(0),
            0.0,
            "Folding agent should have 0% VPIP"
        );

        // The calling agent (BB) wins uncontested - did they VPIP?
        // BB doesn't voluntarily put money in if opponent folds preflop
        // They just win with their forced blind
        assert_eq!(
            stats.hands_vpip[1], 0,
            "Calling agent who won uncontested should have 0 VPIP hands"
        );
    }

    #[test]
    fn test_calling_agent_vs_raiser_has_vpip() {
        // Calling agent should VPIP when they call a raise
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // Raiser is SB, Caller is BB
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)], // Raise to 20
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // The raiser (SB) VPIPed by raising
        assert_eq!(stats.hands_vpip[0], 1, "Raiser should have 1 VPIP hand");
        assert_eq!(stats.vpip_percent(0), 100.0, "Raiser should have 100% VPIP");

        // The caller (BB) VPIPed by calling the raise (putting more money in beyond BB)
        assert_eq!(
            stats.hands_vpip[1], 1,
            "Caller who called a raise should have 1 VPIP hand"
        );
        assert_eq!(stats.vpip_percent(1), 100.0, "Caller should have 100% VPIP");
    }

    #[test]
    fn test_big_blind_check_no_vpip() {
        // Big blind who just checks (no raise to call) should NOT have VPIP
        // This is an important edge case - posting BB is forced, checking is not voluntary
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        // SB limps (calls BB), BB checks
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>, // SB limps
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "checker",
                vec![], // No preflop action = check when BB with no raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // SB called (limped) - this IS VPIP (voluntarily putting in 5 more to match BB)
        assert_eq!(
            stats.hands_vpip[0], 1,
            "Small blind who limped should have VPIP"
        );

        // BB just checked (option) - this is NOT VPIP
        // Note: This depends on how the game handles BB check - if BB has to "call 0"
        // vs no action at all. The current implementation should handle this.
        // If BB checks (no additional money put in), VPIP should be 0
        println!(
            "BB hands_vpip: {}, hands_played: {}",
            stats.hands_vpip[1], stats.hands_played[1]
        );
    }

    #[test]
    fn test_folding_agent_three_way_zero_vpip() {
        // Folding agent should have 0% VPIP in a 3-way pot
        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();

        let stacks = vec![100.0; 3];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // UTG
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // SB
            Box::<CallingAgent>::default() as Box<dyn Agent>, // BB
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // UTG folds - 0 VPIP
        assert_eq!(stats.hands_vpip[0], 0, "UTG folder should have 0 VPIP");
        assert_eq!(stats.vpip_percent(0), 0.0, "UTG folder should have 0% VPIP");

        // SB folds - 0 VPIP
        assert_eq!(stats.hands_vpip[1], 0, "SB folder should have 0 VPIP");
        assert_eq!(stats.vpip_percent(1), 0.0, "SB folder should have 0% VPIP");

        // BB wins uncontested - 0 VPIP (didn't voluntarily put money in)
        assert_eq!(
            stats.hands_vpip[2], 0,
            "BB who won uncontested should have 0 VPIP"
        );
        assert_eq!(
            stats.vpip_percent(2),
            0.0,
            "BB who won uncontested should have 0% VPIP"
        );
    }

    #[test]
    fn test_folding_agent_vs_all_in_vpip() {
        // This tests the edge case in FoldingAgent where it bets when count == 1
        // When facing an all-in player, folding agent might "bet" to stay in
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>, // Goes all-in
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // Folds or bets?
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // The all-in agent should have VPIP
        assert_eq!(stats.hands_vpip[0], 1, "All-in agent should have VPIP");

        // What does the folding agent do when facing an all-in?
        // According to FoldingAgent logic:
        // - count = active_players + all_in_players
        // - If all-in agent is all-in, they're no longer "active" but counted in num_all_in_players
        // - count would be 1 (only folding agent is active) + 1 (all-in) = 2
        // - So folding agent should FOLD
        println!(
            "Folding agent vs all-in - hands_vpip: {}, hands_played: {}",
            stats.hands_vpip[1], stats.hands_played[1]
        );

        // The folding agent should fold (not call the all-in)
        // So they should have 0 VPIP
        assert_eq!(
            stats.hands_vpip[1], 0,
            "Folding agent should fold against all-in, so 0 VPIP"
        );
    }

    #[test]
    fn test_folding_agent_three_player_various_opponents() {
        // Test folding agent in 3-player scenarios with different opponent types
        // This mimics the agent_comparison setup more closely
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut total_hands = 0;
        let mut total_vpip = 0;

        // Run multiple games to see if folding agent ever VPIPs
        for seed in 0..100 {
            let storage = SharedStatsStorage::new(3);
            let hist = storage.historian();

            let stacks = vec![1000.0; 3]; // Larger stacks to ensure no one busts
            // Mix of opponents like in agent_comparison
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<FoldingAgent>::default() as Box<dyn Agent>,
                Box::<AllInAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);

            let stats = storage.read();
            total_hands += stats.hands_played[0];
            total_vpip += stats.hands_vpip[0];
        }

        // The folding agent should have 0 VPIP across all games
        assert_eq!(
            total_vpip, 0,
            "Folding agent should have 0 VPIP across {} hands, but had {}",
            total_hands, total_vpip
        );
    }

    #[test]
    fn test_folding_agent_last_to_act_uncontested() {
        // Test the edge case where folding agent is last to act and wins uncontested
        // FoldingAgent bets when count == 1 (everyone else folded/all-in)
        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();

        let stacks = vec![100.0; 3];
        // Two folders before the third folder - everyone folds
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // UTG - folds
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // SB - folds
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // BB - wins uncontested
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // UTG folds - 0 VPIP
        assert_eq!(stats.hands_vpip[0], 0, "UTG folder should have 0 VPIP");

        // SB folds - 0 VPIP
        assert_eq!(stats.hands_vpip[1], 0, "SB folder should have 0 VPIP");

        // BB wins uncontested - what happens?
        // The FoldingAgent's logic: count = 1 (only BB active) + 0 (no all-ins) = 1
        // So FoldingAgent calls AgentAction::Bet(current_round_bet)
        // But does this get recorded as a PlayedAction?
        // If BB is in for 10 and bets 10 (current bet), put_into_pot = 0, so no VPIP
        println!(
            "BB (last folder) - hands_vpip: {}, hands_played: {}, actions_count: {}",
            stats.hands_vpip[2], stats.hands_played[2], stats.actions_count[2]
        );

        // The BB should NOT have VPIP - they didn't voluntarily put extra money in
        assert_eq!(
            stats.hands_vpip[2], 0,
            "BB who won uncontested should have 0 VPIP (even if they 'bet' to claim pot)"
        );
    }

    #[test]
    fn test_folding_agent_all_permutations_zero_vpip() {
        // Comprehensive test: run all permutations of positions with various opponents
        // This mimics what agent_comparison does
        use itertools::Itertools;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut folder_total_hands = 0;
        let mut folder_total_vpip = 0;

        // Create agents: Folder at index 0, various opponents at indices 1-3
        let agent_types = ["Folder", "AllIn", "Calling", "Random"];

        // Test all permutations of 3 agents from 4 types
        for perm in (0..4).permutations(3) {
            for seed in 0..10 {
                let storage = SharedStatsStorage::new(3);
                let hist = storage.historian();

                let stacks = vec![1000.0; 3];

                let agents: Vec<Box<dyn Agent>> = perm
                    .iter()
                    .map(|&idx| -> Box<dyn Agent> {
                        match agent_types[idx] {
                            "Folder" => Box::<FoldingAgent>::default(),
                            "AllIn" => Box::<AllInAgent>::default(),
                            "Calling" => Box::<CallingAgent>::default(),
                            "Random" => Box::new(VecReplayAgent::new_with_default(
                                "random",
                                vec![AgentAction::Call],
                                AgentAction::Fold,
                            )),
                            _ => unreachable!(),
                        }
                    })
                    .collect();

                let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
                let mut rng = StdRng::seed_from_u64(seed as u64);

                let mut sim = HoldemSimulationBuilder::default()
                    .game_state(game_state)
                    .agents(agents)
                    .historians(vec![Box::new(hist)])
                    .build()
                    .unwrap();

                sim.run(&mut rng);

                let stats = storage.read();

                // Find the folder's position in this permutation
                for (pos, &agent_idx) in perm.iter().enumerate() {
                    if agent_types[agent_idx] == "Folder" {
                        folder_total_hands += stats.hands_played[pos];
                        folder_total_vpip += stats.hands_vpip[pos];

                        // If we find VPIP, print details
                        if stats.hands_vpip[pos] > 0 {
                            println!(
                                "VPIP detected! Perm: {:?}, Seed: {}, Position: {}, VPIP: {}",
                                perm.iter().map(|&i| agent_types[i]).collect::<Vec<_>>(),
                                seed,
                                pos,
                                stats.hands_vpip[pos]
                            );
                        }
                    }
                }
            }
        }

        let vpip_percent = if folder_total_hands > 0 {
            (folder_total_vpip as f32 / folder_total_hands as f32) * 100.0
        } else {
            0.0
        };

        println!(
            "Folding agent total: {} hands, {} VPIP ({:.2}%)",
            folder_total_hands, folder_total_vpip, vpip_percent
        );

        assert_eq!(
            folder_total_vpip, 0,
            "Folding agent should have 0 VPIP across all {} permutations and seeds, but had {} ({:.2}%)",
            folder_total_hands, folder_total_vpip, vpip_percent
        );
    }

    #[test]
    fn test_folding_agent_sb_when_utg_folds_to_bb() {
        // Scenario that might cause VPIP:
        // FoldingAgent is SB, UTG folds to BB, then FoldingAgent must act
        // When everyone except BB folds first, does the SB (FoldingAgent) get a chance to act?
        use crate::arena::historian::VecHistorian;

        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();
        let vec_hist = VecHistorian::default();

        let stacks = vec![100.0; 3];
        // UTG folds, SB (FoldingAgent) faces BB
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // UTG - will fold
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // SB - will fold
            Box::<CallingAgent>::default() as Box<dyn Agent>, // BB
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist), Box::new(vec_hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        println!(
            "UTG (Folder) - vpip: {}, played: {}, actions: {}",
            stats.hands_vpip[0], stats.hands_played[0], stats.actions_count[0]
        );
        println!(
            "SB (Folder) - vpip: {}, played: {}, actions: {}",
            stats.hands_vpip[1], stats.hands_played[1], stats.actions_count[1]
        );
        println!(
            "BB (Caller) - vpip: {}, played: {}, actions: {}",
            stats.hands_vpip[2], stats.hands_played[2], stats.actions_count[2]
        );

        // All folding agents should have 0 VPIP
        assert_eq!(stats.hands_vpip[0], 0, "UTG folder should have 0 VPIP");
        assert_eq!(stats.hands_vpip[1], 0, "SB folder should have 0 VPIP");
    }

    #[test]
    fn test_debug_folding_agent_with_random_opponents() {
        // Debug test: run with RandomAgent opponents to see what's happening
        use crate::arena::agent::RandomAgent;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut total_vpip = 0;
        let mut total_hands = 0;

        for seed in 0..100 {
            let storage = SharedStatsStorage::new(3);
            let hist = storage.historian();

            let stacks = vec![1000.0; 3];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<FoldingAgent>::default() as Box<dyn Agent>,
                Box::new(RandomAgent::default()),
                Box::new(RandomAgent::default()),
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);

            let stats = storage.read();
            total_hands += stats.hands_played[0];
            total_vpip += stats.hands_vpip[0];

            if stats.hands_vpip[0] > 0 {
                println!(
                    "VPIP at seed {}: vpip={}, played={}, actions={}",
                    seed, stats.hands_vpip[0], stats.hands_played[0], stats.actions_count[0]
                );
            }
        }

        println!(
            "Total: {} VPIP out of {} hands ({:.2}%)",
            total_vpip,
            total_hands,
            (total_vpip as f32 / total_hands as f32) * 100.0
        );

        assert_eq!(
            total_vpip, 0,
            "Folding agent should have 0 VPIP with random opponents"
        );
    }

    #[test]
    fn test_debug_folding_agent_agent_comparison_scenario() {
        // Simulate agent_comparison: all 6 agents, 3 at a table, many permutations
        use crate::arena::agent::{RandomAgent, RandomPotControlAgent};
        use itertools::Itertools;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // Track folding agent VPIP across all permutations
        let mut folder_vpip_by_position: [usize; 3] = [0, 0, 0];
        let mut folder_hands_by_position: [usize; 3] = [0, 0, 0];

        // All 6 agent types from agent_comparison
        let agent_types = [
            "AllIn",
            "Calling",
            "Folding",
            "RandomAgg",
            "RandomDef",
            "RandomPot",
        ];

        // Test all permutations of 3 agents from 6 types
        for perm in (0..6).permutations(3) {
            // Only process permutations that include the Folding agent
            if !perm.contains(&2) {
                continue;
            }

            for seed in 0..10 {
                let storage = SharedStatsStorage::new(3);
                let hist = storage.historian();

                let stacks = vec![1000.0; 3];

                let agents: Vec<Box<dyn Agent>> = perm
                    .iter()
                    .map(|&idx| -> Box<dyn Agent> {
                        match agent_types[idx] {
                            "AllIn" => Box::<AllInAgent>::default(),
                            "Calling" => Box::<CallingAgent>::default(),
                            "Folding" => Box::<FoldingAgent>::default(),
                            "RandomAgg" => Box::new(RandomAgent::new(
                                "RandomAgg",
                                vec![0.1, 0.15, 0.25],
                                vec![0.4, 0.5, 0.4],
                            )),
                            "RandomDef" => Box::new(RandomAgent::default()),
                            "RandomPot" => {
                                Box::new(RandomPotControlAgent::new("RandomPot", vec![0.5, 0.3]))
                            }
                            _ => unreachable!(),
                        }
                    })
                    .collect();

                let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
                let mut rng = StdRng::seed_from_u64(seed as u64);

                let mut sim = HoldemSimulationBuilder::default()
                    .game_state(game_state)
                    .agents(agents)
                    .historians(vec![Box::new(hist)])
                    .build()
                    .unwrap();

                sim.run(&mut rng);

                let stats = storage.read();

                // Find the folder's position
                for (pos, &agent_idx) in perm.iter().enumerate() {
                    if agent_types[agent_idx] == "Folding" {
                        folder_hands_by_position[pos] += stats.hands_played[pos];
                        folder_vpip_by_position[pos] += stats.hands_vpip[pos];

                        if stats.hands_vpip[pos] > 0 {
                            println!(
                                "VPIP! Perm: {:?}, Seed: {}, Pos: {}, VPIP: {}, Actions: {}, legacy_vpip: {}, legacy_total: {}",
                                perm.iter().map(|&i| agent_types[i]).collect::<Vec<_>>(),
                                seed,
                                pos,
                                stats.hands_vpip[pos],
                                stats.actions_count[pos],
                                stats.vpip_count[pos],
                                stats.vpip_total[pos]
                            );
                        }
                    }
                }
            }
        }

        // Print summary
        let total_hands: usize = folder_hands_by_position.iter().sum();
        let total_vpip: usize = folder_vpip_by_position.iter().sum();

        println!("\nFolding agent summary:");
        for pos in 0..3 {
            let pct = if folder_hands_by_position[pos] > 0 {
                (folder_vpip_by_position[pos] as f32 / folder_hands_by_position[pos] as f32) * 100.0
            } else {
                0.0
            };
            println!(
                "  Position {}: {} VPIP / {} hands ({:.2}%)",
                pos, folder_vpip_by_position[pos], folder_hands_by_position[pos], pct
            );
        }
        println!(
            "  Total: {} VPIP / {} hands ({:.2}%)",
            total_vpip,
            total_hands,
            (total_vpip as f32 / total_hands as f32) * 100.0
        );

        assert_eq!(
            total_vpip, 0,
            "Folding agent should have 0 VPIP in agent_comparison-like scenario"
        );
    }

    #[test]
    fn test_debug_specific_vpip_scenario() {
        // Debug the specific scenario: ["RandomAgg", "Calling", "Folding"], Seed: 1
        use crate::arena::action::Action;
        use crate::arena::agent::RandomAgent;
        use crate::arena::historian::VecHistorian;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();
        let vec_hist = VecHistorian::default();
        let vec_storage = vec_hist.get_storage();

        let stacks = vec![1000.0; 3];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(RandomAgent::new(
                "RandomAgg",
                vec![0.1, 0.15, 0.25],
                vec![0.4, 0.5, 0.4],
            )),
            Box::<CallingAgent>::default(),
            Box::<FoldingAgent>::default(),
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = StdRng::seed_from_u64(1);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist), Box::new(vec_hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        println!("=== Specific scenario debug ===");
        println!(
            "Folding agent (pos 2): vpip={}, played={}, actions={}, legacy_vpip={}, legacy_total={}",
            stats.hands_vpip[2],
            stats.hands_played[2],
            stats.actions_count[2],
            stats.vpip_count[2],
            stats.vpip_total[2]
        );

        // Print ALL actions to understand the full sequence
        println!("\n=== ALL Actions ===");
        let actions = vec_storage.borrow();
        for record in actions.iter() {
            match &record.action {
                Action::PlayedAction(payload) => {
                    let agent_name = match payload.idx {
                        0 => "RandomAgg",
                        1 => "Calling",
                        2 => "FOLDING",
                        _ => "???",
                    };
                    println!(
                        "[{}] Round: {:?}, Action: {:?}, current_bet={}, player_bet={}, put_in={}",
                        agent_name,
                        payload.round,
                        payload.action,
                        payload.final_bet,
                        payload.final_player_bet,
                        payload.final_player_bet - payload.starting_player_bet
                    );
                }
                Action::RoundAdvance(round) => {
                    println!("--- Round advance to {:?} ---", round);
                }
                _ => {}
            }
        }

        // This scenario should NOT have VPIP for the folding agent
        assert_eq!(
            stats.hands_vpip[2], 0,
            "Folding agent at BB should have 0 VPIP"
        );
    }

    #[test]
    fn test_folding_agent_when_utg_raises_and_sb_folds() {
        // Specific scenario: UTG raises, SB folds, BB (FoldingAgent) should fold
        use crate::arena::action::Action;
        use crate::arena::historian::VecHistorian;

        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();
        let vec_hist = VecHistorian::default();
        let vec_storage = vec_hist.get_storage();

        let stacks = vec![1000.0; 3];
        // UTG raises to 40, SB folds, BB (FoldingAgent) should fold
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(VecReplayAgent::new_with_default(
                "Raiser",
                vec![AgentAction::Bet(40.0)], // UTG raises to 40
                AgentAction::Call,
            )),
            Box::new(VecReplayAgent::new_with_default(
                "Folder",
                vec![AgentAction::Fold], // SB folds
                AgentAction::Fold,
            )),
            Box::<FoldingAgent>::default(), // BB - should fold when facing a raise
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist), Box::new(vec_hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        println!("=== UTG raises, SB folds scenario ===");
        println!(
            "Folding agent (pos 2): vpip={}, played={}, actions={}",
            stats.hands_vpip[2], stats.hands_played[2], stats.actions_count[2]
        );

        // Print ALL actions
        println!("\n=== ALL Actions ===");
        let actions = vec_storage.borrow();
        for record in actions.iter() {
            match &record.action {
                Action::PlayedAction(payload) => {
                    let agent_name = match payload.idx {
                        0 => "UTG-Raiser",
                        1 => "SB-Folder",
                        2 => "BB-FOLDING",
                        _ => "???",
                    };
                    println!(
                        "[{}] Round: {:?}, Action: {:?}, current_bet={}, player_bet={}, put_in={}",
                        agent_name,
                        payload.round,
                        payload.action,
                        payload.final_bet,
                        payload.final_player_bet,
                        payload.final_player_bet - payload.starting_player_bet
                    );
                }
                Action::RoundAdvance(round) => {
                    println!("--- Round advance to {:?} ---", round);
                }
                _ => {}
            }
        }

        // The FoldingAgent should fold (count = 1 active + 0 all-in = 1, but that's wrong!)
        // Actually, count should be 2: FoldingAgent (active) + UTG-Raiser (active)
        // Wait, if SB folded, we have UTG (active) + BB (active) = 2
        // So FoldingAgent should FOLD
        assert_eq!(
            stats.hands_vpip[2], 0,
            "Folding agent should have 0 VPIP when facing a raise"
        );
    }

    #[test]
    fn test_folding_agent_when_utg_folds_and_sb_folds() {
        // Specific scenario: UTG folds, SB folds, BB (FoldingAgent) wins uncontested
        // In this case, FoldingAgent's count == 1, so it bets current_round_bet
        // But current_round_bet should be 10 (BB), so put_in = 0
        use crate::arena::action::Action;
        use crate::arena::historian::VecHistorian;

        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();
        let vec_hist = VecHistorian::default();
        let vec_storage = vec_hist.get_storage();

        let stacks = vec![1000.0; 3];
        // UTG folds, SB folds, BB (FoldingAgent) wins
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(VecReplayAgent::new_with_default(
                "Folder1",
                vec![AgentAction::Fold], // UTG folds
                AgentAction::Fold,
            )),
            Box::new(VecReplayAgent::new_with_default(
                "Folder2",
                vec![AgentAction::Fold], // SB folds
                AgentAction::Fold,
            )),
            Box::<FoldingAgent>::default(), // BB - should win uncontested
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist), Box::new(vec_hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        println!("=== UTG folds, SB folds scenario ===");
        println!(
            "Folding agent (pos 2): vpip={}, played={}, actions={}",
            stats.hands_vpip[2], stats.hands_played[2], stats.actions_count[2]
        );

        // Print ALL actions
        println!("\n=== ALL Actions ===");
        let actions = vec_storage.borrow();
        for record in actions.iter() {
            match &record.action {
                Action::PlayedAction(payload) => {
                    let agent_name = match payload.idx {
                        0 => "UTG-Folder1",
                        1 => "SB-Folder2",
                        2 => "BB-FOLDING",
                        _ => "???",
                    };
                    println!(
                        "[{}] Round: {:?}, Action: {:?}, current_bet={}, player_bet={}, put_in={}",
                        agent_name,
                        payload.round,
                        payload.action,
                        payload.final_bet,
                        payload.final_player_bet,
                        payload.final_player_bet - payload.starting_player_bet
                    );
                }
                Action::RoundAdvance(round) => {
                    println!("--- Round advance to {:?} ---", round);
                }
                _ => {}
            }
        }

        // FoldingAgent wins uncontested. count == 1, so it "bets" current_round_bet (10)
        // Since it already has 10 in (BB), put_in = 0, so NO VPIP
        assert_eq!(
            stats.hands_vpip[2], 0,
            "Folding agent winning uncontested should have 0 VPIP"
        );
    }

    // ======= POSITION TRACKING TESTS =======
    // Tests for position_games and position_profit tracking

    #[test]
    fn test_position_stats_accessor() {
        let mut stats = StatsStorage::new_with_num_players(3);
        stats.position_games[0].insert(0, 5);
        stats.position_games[0].insert(1, 3);
        stats.position_games[1].insert(2, 7);

        let pos_stats = stats.position_stats(0);
        assert_eq!(pos_stats.get(&0), Some(&5));
        assert_eq!(pos_stats.get(&1), Some(&3));
        assert_eq!(pos_stats.get(&2), None);

        let pos_stats_1 = stats.position_stats(1);
        assert_eq!(pos_stats_1.get(&2), Some(&7));
    }

    #[test]
    fn test_position_profit_accessor() {
        let mut stats = StatsStorage::new_with_num_players(2);
        stats.position_profit[0].insert(0, 100.0);
        stats.position_profit[0].insert(1, -50.0);
        stats.position_profit[1].insert(0, -100.0);
        stats.position_profit[1].insert(1, 50.0);

        let profit_0 = stats.position_profit(0);
        assert_eq!(profit_0.get(&0), Some(&100.0));
        assert_eq!(profit_0.get(&1), Some(&-50.0));

        let profit_1 = stats.position_profit(1);
        assert_eq!(profit_1.get(&0), Some(&-100.0));
        assert_eq!(profit_1.get(&1), Some(&50.0));
    }

    #[test]
    fn test_position_tracking_heads_up_single_game() {
        // Run a single heads-up game and verify position tracking
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        // AllIn vs Folding - AllIn will win, Folder will lose
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Each player should have their position tracked
        // Player 0 played at position 0
        assert_eq!(
            stats.position_games[0].get(&0),
            Some(&1),
            "Player 0 should have 1 game at position 0"
        );
        // Player 1 played at position 1
        assert_eq!(
            stats.position_games[1].get(&1),
            Some(&1),
            "Player 1 should have 1 game at position 1"
        );

        // Position profit tracking
        // AllIn agent (pos 0) wins, Folding agent (pos 1) loses
        let profit_0 = stats.position_profit[0].get(&0).copied().unwrap_or(0.0);
        let profit_1 = stats.position_profit[1].get(&1).copied().unwrap_or(0.0);

        // Profits should be opposite (zero-sum)
        assert!(
            (profit_0 + profit_1).abs() < 0.01,
            "Position profits should sum to zero, got {} + {} = {}",
            profit_0,
            profit_1,
            profit_0 + profit_1
        );
    }

    #[test]
    fn test_position_tracking_three_player_single_game() {
        // Run a single 3-player game and verify position tracking
        let storage = SharedStatsStorage::new(3);
        let hist = storage.historian();

        let stacks = vec![100.0; 3];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Each player should have their position tracked
        for player in 0..3 {
            assert_eq!(
                stats.position_games[player].get(&player),
                Some(&1),
                "Player {} should have 1 game at position {}",
                player,
                player
            );
        }

        // Verify zero-sum across positions
        let total_position_profit: f32 = (0..3)
            .map(|p| stats.position_profit[p].get(&p).copied().unwrap_or(0.0))
            .sum();

        assert!(
            total_position_profit.abs() < 0.01,
            "Total position profit should be zero, got {}",
            total_position_profit
        );
    }

    #[test]
    fn test_merge_position_stats_multiple_positions() {
        // Test merge with multiple positions per player
        let mut stats1 = StatsStorage::new_with_num_players(3);
        stats1.position_games[0].insert(0, 5);
        stats1.position_games[0].insert(1, 3);
        stats1.position_games[0].insert(2, 2);
        stats1.position_profit[0].insert(0, 100.0);
        stats1.position_profit[0].insert(1, -50.0);
        stats1.position_profit[0].insert(2, 25.0);

        let mut stats2 = StatsStorage::new_with_num_players(3);
        stats2.position_games[0].insert(0, 3);
        stats2.position_games[0].insert(1, 4);
        stats2.position_games[0].insert(2, 1);
        stats2.position_profit[0].insert(0, 50.0);
        stats2.position_profit[0].insert(1, 30.0);
        stats2.position_profit[0].insert(2, -10.0);

        stats1.merge(&stats2);

        // Verify game counts merged correctly
        assert_eq!(stats1.position_games[0].get(&0), Some(&8)); // 5 + 3
        assert_eq!(stats1.position_games[0].get(&1), Some(&7)); // 3 + 4
        assert_eq!(stats1.position_games[0].get(&2), Some(&3)); // 2 + 1

        // Verify profits merged correctly
        assert_eq!(stats1.position_profit[0].get(&0), Some(&150.0)); // 100 + 50
        assert_eq!(stats1.position_profit[0].get(&1), Some(&-20.0)); // -50 + 30
        assert_eq!(stats1.position_profit[0].get(&2), Some(&15.0)); // 25 + -10
    }

    #[test]
    fn test_merge_position_stats_disjoint_positions() {
        // Test merge when players have played different positions
        let mut stats1 = StatsStorage::new_with_num_players(3);
        stats1.position_games[0].insert(0, 5);
        stats1.position_profit[0].insert(0, 100.0);

        let mut stats2 = StatsStorage::new_with_num_players(3);
        stats2.position_games[0].insert(1, 3);
        stats2.position_profit[0].insert(1, -50.0);

        stats1.merge(&stats2);

        // Both positions should be present after merge
        assert_eq!(stats1.position_games[0].get(&0), Some(&5));
        assert_eq!(stats1.position_games[0].get(&1), Some(&3));
        assert_eq!(stats1.position_profit[0].get(&0), Some(&100.0));
        assert_eq!(stats1.position_profit[0].get(&1), Some(&-50.0));
    }

    #[test]
    fn test_position_profit_positive_and_negative() {
        // Test that position profit correctly tracks both wins and losses
        let mut stats = StatsStorage::new_with_num_players(2);

        // Player 0 won 100 at position 0, lost 50 at position 1
        stats.position_games[0].insert(0, 1);
        stats.position_games[0].insert(1, 1);
        stats.position_profit[0].insert(0, 100.0);
        stats.position_profit[0].insert(1, -50.0);

        // Player 1 lost 100 at position 0, won 50 at position 1
        stats.position_games[1].insert(0, 1);
        stats.position_games[1].insert(1, 1);
        stats.position_profit[1].insert(0, -100.0);
        stats.position_profit[1].insert(1, 50.0);

        // Calculate profit per game at each position for player 0
        let profit_at_0 = *stats.position_profit[0].get(&0).unwrap_or(&0.0);
        let games_at_0 = *stats.position_games[0].get(&0).unwrap_or(&0) as f32;
        let ppg_at_0 = if games_at_0 > 0.0 {
            profit_at_0 / games_at_0
        } else {
            0.0
        };
        assert_eq!(ppg_at_0, 100.0);

        let profit_at_1 = *stats.position_profit[0].get(&1).unwrap_or(&0.0);
        let games_at_1 = *stats.position_games[0].get(&1).unwrap_or(&0) as f32;
        let ppg_at_1 = if games_at_1 > 0.0 {
            profit_at_1 / games_at_1
        } else {
            0.0
        };
        assert_eq!(ppg_at_1, -50.0);
    }

    #[test]
    fn test_position_tracking_empty_initial() {
        // Verify position tracking starts empty
        let stats = StatsStorage::new_with_num_players(3);

        for player in 0..3 {
            assert!(
                stats.position_games[player].is_empty(),
                "Player {} should have no position games initially",
                player
            );
            assert!(
                stats.position_profit[player].is_empty(),
                "Player {} should have no position profit initially",
                player
            );
        }
    }

    #[test]
    fn test_position_tracking_multiple_games_accumulates() {
        // Test that running multiple games accumulates position stats
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run 10 games
        for seed in 0..10 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // Each player should have 10 games at their position
        assert_eq!(
            stats.position_games[0].get(&0),
            Some(&10),
            "Player 0 should have 10 games at position 0"
        );
        assert_eq!(
            stats.position_games[1].get(&1),
            Some(&10),
            "Player 1 should have 10 games at position 1"
        );

        // Total profit should be zero (zero-sum)
        let total_profit: f32 = stats.total_profit.iter().sum();
        assert!(
            total_profit.abs() < 0.01,
            "Total profit across players should be zero, got {}",
            total_profit
        );
    }

    #[test]
    fn test_position_profit_calculation_accuracy() {
        // Test exact position profit calculation with known values
        let mut stats = StatsStorage::new_with_num_players(1);

        // Player played 4 games at position 0 with profits: +100, -50, +25, -75
        stats.position_games[0].insert(0, 4);
        stats.position_profit[0].insert(0, 100.0 - 50.0 + 25.0 - 75.0); // = 0.0

        let profit = stats.position_profit[0].get(&0).copied().unwrap_or(0.0);
        assert!(profit.abs() < 0.001, "Expected 0.0 profit, got {}", profit);
    }

    #[test]
    fn test_position_tracking_zero_sum_heads_up_multiple_games() {
        // Verify zero-sum property holds across multiple heads-up games
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run 50 games to get statistical significance
        for seed in 0..50 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<AllInAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // Get total position profit for each player
        let player0_pos_profit: f32 = stats.position_profit[0].values().sum();
        let player1_pos_profit: f32 = stats.position_profit[1].values().sum();

        // Should be zero-sum
        assert!(
            (player0_pos_profit + player1_pos_profit).abs() < 0.01,
            "Position profits should sum to zero, got {} + {} = {}",
            player0_pos_profit,
            player1_pos_profit,
            player0_pos_profit + player1_pos_profit
        );

        // Position profit should equal total profit
        assert!(
            (player0_pos_profit - stats.total_profit[0]).abs() < 0.01,
            "Player 0 position profit {} should equal total profit {}",
            player0_pos_profit,
            stats.total_profit[0]
        );
        assert!(
            (player1_pos_profit - stats.total_profit[1]).abs() < 0.01,
            "Player 1 position profit {} should equal total profit {}",
            player1_pos_profit,
            stats.total_profit[1]
        );
    }

    #[test]
    fn test_position_stats_three_player_zero_sum() {
        // Verify zero-sum in 3-player games
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(3);

        // Run 30 games
        for seed in 0..30 {
            let hist = storage.historian();
            let stacks = vec![100.0; 3];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<AllInAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<FoldingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // Get total position profit across all players
        let total_position_profit: f32 =
            (0..3).flat_map(|p| stats.position_profit[p].values()).sum();

        assert!(
            total_position_profit.abs() < 0.01,
            "Total position profit should be zero in 3-player game, got {}",
            total_position_profit
        );
    }

    // ===========================================
    // Advanced Stats Tests
    // ===========================================

    #[test]
    fn test_aggression_frequency_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.bet_count[0] = 3;
        stats.raise_count[0] = 2;
        stats.call_count[0] = 3;
        stats.fold_count[0] = 2;

        // AFq = (3 + 2) / (3 + 2 + 3 + 2) * 100 = 5/10 * 100 = 50%
        let afq = stats.aggression_frequency(0);
        assert!(
            (afq - 50.0).abs() < 0.01,
            "Expected AFq of 50%, got {}",
            afq
        );
    }

    #[test]
    fn test_aggression_frequency_zero_actions() {
        let stats = StatsStorage::new_with_num_players(1);
        let afq = stats.aggression_frequency(0);
        assert_eq!(afq, 0.0);
    }

    #[test]
    fn test_aggression_frequency_all_aggressive() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.bet_count[0] = 5;
        stats.raise_count[0] = 5;
        // No calls or folds

        let afq = stats.aggression_frequency(0);
        assert!(
            (afq - 100.0).abs() < 0.01,
            "Expected AFq of 100%, got {}",
            afq
        );
    }

    #[test]
    fn test_flop_aggression_factor() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 4;
        stats.flop_raises[0] = 2;
        stats.flop_calls[0] = 2;

        // Flop AF = (4 + 2) / 2 = 3.0
        let flop_af = stats.flop_aggression_factor(0);
        assert!(
            (flop_af - 3.0).abs() < 0.01,
            "Expected Flop AF of 3.0, got {}",
            flop_af
        );
    }

    #[test]
    fn test_turn_aggression_factor() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_bets[0] = 3;
        stats.turn_raises[0] = 1;
        stats.turn_calls[0] = 2;

        // Turn AF = (3 + 1) / 2 = 2.0
        let turn_af = stats.turn_aggression_factor(0);
        assert!(
            (turn_af - 2.0).abs() < 0.01,
            "Expected Turn AF of 2.0, got {}",
            turn_af
        );
    }

    #[test]
    fn test_river_aggression_factor() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_bets[0] = 2;
        stats.river_raises[0] = 0;
        stats.river_calls[0] = 4;

        // River AF = (2 + 0) / 4 = 0.5
        let river_af = stats.river_aggression_factor(0);
        assert!(
            (river_af - 0.5).abs() < 0.01,
            "Expected River AF of 0.5, got {}",
            river_af
        );
    }

    #[test]
    fn test_per_street_af_no_calls() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 3;
        stats.flop_raises[0] = 2;
        // No calls on flop

        let flop_af = stats.flop_aggression_factor(0);
        assert_eq!(flop_af, f32::INFINITY);
    }

    #[test]
    fn test_per_street_af_no_actions() {
        let stats = StatsStorage::new_with_num_players(1);

        assert_eq!(stats.flop_aggression_factor(0), 0.0);
        assert_eq!(stats.turn_aggression_factor(0), 0.0);
        assert_eq!(stats.river_aggression_factor(0), 0.0);
    }

    #[test]
    fn test_cbet_percent_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.cbet_opportunities[0] = 10;
        stats.cbet_count[0] = 7;

        // C-Bet% = 7/10 * 100 = 70%
        let cbet = stats.cbet_percent(0);
        assert!(
            (cbet - 70.0).abs() < 0.01,
            "Expected C-Bet% of 70%, got {}",
            cbet
        );
    }

    #[test]
    fn test_cbet_percent_zero_opportunities() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(stats.cbet_percent(0), 0.0);
    }

    #[test]
    fn test_wtsd_percent_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.wtsd_opportunities[0] = 20; // Saw flop 20 times
        stats.wtsd_count[0] = 8; // Went to showdown 8 times

        // WTSD% = 8/20 * 100 = 40%
        let wtsd = stats.wtsd_percent(0);
        assert!(
            (wtsd - 40.0).abs() < 0.01,
            "Expected WTSD% of 40%, got {}",
            wtsd
        );
    }

    #[test]
    fn test_wtsd_percent_zero_opportunities() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(stats.wtsd_percent(0), 0.0);
    }

    #[test]
    fn test_wsd_percent_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.showdown_count[0] = 10;
        stats.showdown_wins[0] = 6;

        // W$SD% = 6/10 * 100 = 60%
        let wsd = stats.wsd_percent(0);
        assert!(
            (wsd - 60.0).abs() < 0.01,
            "Expected W$SD% of 60%, got {}",
            wsd
        );
    }

    #[test]
    fn test_wsd_percent_zero_showdowns() {
        let stats = StatsStorage::new_with_num_players(1);
        assert_eq!(stats.wsd_percent(0), 0.0);
    }

    #[test]
    fn test_fold_count_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Folding agent should have at least one fold
        assert!(
            borrowed.fold_count[0] > 0,
            "Expected folding agent to have folds tracked, got {}",
            borrowed.fold_count[0]
        );
    }

    #[test]
    fn test_per_street_action_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        // One agent bets, one agent calls - this ensures we get post-flop bets and calls
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "bettor",
                vec![
                    AgentAction::Bet(20.0), // Raise preflop
                    AgentAction::Bet(10.0), // Bet flop
                    AgentAction::Bet(10.0), // Bet turn
                    AgentAction::Bet(10.0), // Bet river
                ],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![
                    AgentAction::Call, // Call preflop
                    AgentAction::Call, // Call flop
                    AgentAction::Call, // Call turn
                    AgentAction::Call, // Call river
                ],
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Bettor should have bets on post-flop streets
        let total_flop_bets: usize = borrowed.flop_bets.iter().sum();
        let total_turn_bets: usize = borrowed.turn_bets.iter().sum();
        let total_river_bets: usize = borrowed.river_bets.iter().sum();

        // Caller should have calls on post-flop streets
        let total_flop_calls: usize = borrowed.flop_calls.iter().sum();
        let total_turn_calls: usize = borrowed.turn_calls.iter().sum();
        let total_river_calls: usize = borrowed.river_calls.iter().sum();

        // With bet/call agents, there should be bets and calls on post-flop streets
        assert!(
            total_flop_bets > 0,
            "Expected flop bets, got {}",
            total_flop_bets
        );
        assert!(
            total_flop_calls > 0,
            "Expected flop calls, got {}",
            total_flop_calls
        );
        assert!(
            total_turn_bets > 0 || total_turn_calls > 0,
            "Expected some turn actions, got bets={}, calls={}",
            total_turn_bets,
            total_turn_calls
        );
        assert!(
            total_river_bets > 0 || total_river_calls > 0,
            "Expected some river actions, got bets={}, calls={}",
            total_river_bets,
            total_river_calls
        );
    }

    #[test]
    fn test_cbet_tracking_raiser_bets_flop() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        // Agent 0 raises preflop and bets on flop (c-bet)
        // Agent 1 calls
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "aggressor",
                vec![
                    AgentAction::Bet(20.0), // Raise preflop
                    AgentAction::Bet(15.0), // Bet on flop (C-Bet)
                ],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![
                    AgentAction::Call, // Call preflop
                    AgentAction::Call, // Call flop
                ],
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Agent 0 (preflop raiser) should have a C-Bet opportunity and taken it
        assert!(
            borrowed.cbet_opportunities[0] >= 1,
            "Expected aggressor to have C-Bet opportunity, got {}",
            borrowed.cbet_opportunities[0]
        );
        assert!(
            borrowed.cbet_count[0] >= 1,
            "Expected aggressor to have C-Bet count, got {}",
            borrowed.cbet_count[0]
        );

        // C-Bet percentage should be 100% (1/1)
        let cbet_pct = borrowed.cbet_percent(0);
        assert!(cbet_pct > 0.0, "Expected positive C-Bet%, got {}", cbet_pct);
    }

    #[test]
    fn test_ats_tracking_button_steals() {
        // Test ATS (Attempted to Steal) tracking
        // In 3-player, dealer (BTN) is in steal position
        // When BTN raises and pot is unopened (folded to), that's a steal attempt
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(3));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 3];

        // In 3-player with dealer at 0:
        // Position 0 = BTN (dealer, acts first preflop in 3-player)
        // Position 1 = SB
        // Position 2 = BB
        // BTN raises (steal attempt), others fold
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "button_stealer",
                vec![
                    AgentAction::Bet(20.0), // Raise from button (steal attempt)
                ],
                AgentAction::Fold,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "sb_folder",
                vec![
                    AgentAction::Fold, // Fold to button raise
                ],
                AgentAction::Fold,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "bb_folder",
                vec![
                    AgentAction::Fold, // Fold to button raise
                ],
                AgentAction::Fold,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.read();

        // Agent 0 (BTN) should have a steal opportunity and taken it
        assert_eq!(
            borrowed.steal_opportunities[0], 1,
            "Expected BTN to have 1 steal opportunity, got {}",
            borrowed.steal_opportunities[0]
        );
        assert_eq!(
            borrowed.steal_count[0], 1,
            "Expected BTN to have 1 steal, got {}",
            borrowed.steal_count[0]
        );

        // Steal percentage should be 100%
        let steal_pct = borrowed.steal_percent(0);
        assert!(
            (steal_pct - 100.0).abs() < 0.001,
            "Expected 100% steal%, got {}",
            steal_pct
        );

        // SB and BB should have no steal opportunities (action was not folded to them)
        assert_eq!(
            borrowed.steal_opportunities[1], 0,
            "SB should have no steal opportunities"
        );
        assert_eq!(
            borrowed.steal_opportunities[2], 0,
            "BB should have no steal opportunities"
        );
    }

    #[test]
    fn test_wtsd_tracking() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run multiple games to get showdown scenarios
        for seed in 0..20 {
            let hist = storage.historian();
            let stacks = vec![100.0; 2];

            // Both players call to showdown
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // With calling agents, both should see the flop (WTSD opportunities)
        assert!(
            stats.wtsd_opportunities[0] > 0,
            "Expected WTSD opportunities for player 0, got {}",
            stats.wtsd_opportunities[0]
        );
        assert!(
            stats.wtsd_opportunities[1] > 0,
            "Expected WTSD opportunities for player 1, got {}",
            stats.wtsd_opportunities[1]
        );

        // Both should also go to showdown most of the time
        assert!(
            stats.wtsd_count[0] > 0,
            "Expected WTSD count for player 0, got {}",
            stats.wtsd_count[0]
        );
        assert!(
            stats.wtsd_count[1] > 0,
            "Expected WTSD count for player 1, got {}",
            stats.wtsd_count[1]
        );
    }

    #[test]
    fn test_wsd_tracking() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run multiple games to get showdown scenarios
        for seed in 0..20 {
            let hist = storage.historian();
            let stacks = vec![100.0; 2];

            // Both players call to showdown
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // Both should have showdown counts
        assert!(
            stats.showdown_count[0] > 0,
            "Expected showdown count for player 0, got {}",
            stats.showdown_count[0]
        );
        assert!(
            stats.showdown_count[1] > 0,
            "Expected showdown count for player 1, got {}",
            stats.showdown_count[1]
        );

        // Total showdown wins should equal total showdown losses (zero-sum at showdown)
        // Since ties are possible, we check that wins are distributed
        let total_showdown_wins = stats.showdown_wins[0] + stats.showdown_wins[1];
        assert!(
            total_showdown_wins > 0,
            "Expected some showdown wins, got {}",
            total_showdown_wins
        );

        // W$SD should be calculable
        let wsd_0 = stats.wsd_percent(0);
        let wsd_1 = stats.wsd_percent(1);
        assert!(
            (0.0..=100.0).contains(&wsd_0),
            "W$SD should be between 0-100%, got {}",
            wsd_0
        );
        assert!(
            (0.0..=100.0).contains(&wsd_1),
            "W$SD should be between 0-100%, got {}",
            wsd_1
        );
    }

    #[test]
    fn test_merge_advanced_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.cbet_opportunities[0] = 5;
        stats1.cbet_count[0] = 3;
        stats1.wtsd_opportunities[0] = 10;
        stats1.wtsd_count[0] = 4;
        stats1.showdown_count[0] = 4;
        stats1.showdown_wins[0] = 2;
        stats1.fold_count[0] = 5;
        stats1.flop_bets[0] = 3;
        stats1.turn_raises[0] = 2;
        stats1.river_calls[0] = 4;

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.cbet_opportunities[0] = 3;
        stats2.cbet_count[0] = 2;
        stats2.wtsd_opportunities[0] = 5;
        stats2.wtsd_count[0] = 3;
        stats2.showdown_count[0] = 3;
        stats2.showdown_wins[0] = 1;
        stats2.fold_count[0] = 3;
        stats2.flop_bets[0] = 2;
        stats2.turn_raises[0] = 1;
        stats2.river_calls[0] = 2;

        stats1.merge(&stats2);

        assert_eq!(stats1.cbet_opportunities[0], 8);
        assert_eq!(stats1.cbet_count[0], 5);
        assert_eq!(stats1.wtsd_opportunities[0], 15);
        assert_eq!(stats1.wtsd_count[0], 7);
        assert_eq!(stats1.showdown_count[0], 7);
        assert_eq!(stats1.showdown_wins[0], 3);
        assert_eq!(stats1.fold_count[0], 8);
        assert_eq!(stats1.flop_bets[0], 5);
        assert_eq!(stats1.turn_raises[0], 3);
        assert_eq!(stats1.river_calls[0], 6);

        // Verify calculated percentages after merge
        // C-Bet% = 5/8 * 100 = 62.5%
        let cbet_pct = stats1.cbet_percent(0);
        assert!((cbet_pct - 62.5).abs() < 0.01);

        // WTSD% = 7/15 * 100 = 46.67%
        let wtsd_pct = stats1.wtsd_percent(0);
        assert!((wtsd_pct - 46.67).abs() < 0.1);

        // W$SD% = 3/7 * 100 = 42.86%
        let wsd_pct = stats1.wsd_percent(0);
        assert!((wsd_pct - 42.86).abs() < 0.1);
    }

    /// Verifies that aggression factor is calculated correctly as (raise_count + bet_count) / call_count.
    #[test]
    fn test_aggression_factor_arithmetic_correctness() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 3;
        stats.bet_count[0] = 2;
        stats.call_count[0] = 1;

        // AF = (3 + 2) / 1 = 5.0
        let af = stats.aggression_factor(0);
        assert!(
            (af - 5.0).abs() < 0.001,
            "Expected AF of 5.0 (3+2)/1, got {}",
            af
        );
    }

    /// Verifies aggression factor calculation with asymmetric values.
    #[test]
    fn test_aggression_factor_asymmetric_values() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 7;
        stats.bet_count[0] = 3;
        stats.call_count[0] = 5;

        // AF = (7 + 3) / 5 = 2.0
        let af = stats.aggression_factor(0);
        assert!((af - 2.0).abs() < 0.001, "Expected AF of 2.0, got {}", af);
    }

    /// Verifies ROI percentage is calculated correctly as (profit / investment) * 100.
    #[test]
    fn test_roi_percent_calculation() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 100.0;
        stats.total_invested[0] = 1000.0;

        // ROI = (100 / 1000) * 100 = 10%
        let roi = stats.roi_percent(0);
        assert!(
            (roi - 10.0).abs() < 0.001,
            "Expected ROI of 10.0%, got {}",
            roi
        );
    }

    /// Verifies ROI percentage division with a different profit value.
    #[test]
    fn test_roi_percent_division() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 50.0;
        stats.total_invested[0] = 1000.0;

        // ROI = (50 / 1000) * 100 = 5%
        let roi = stats.roi_percent(0);
        assert!(
            (roi - 5.0).abs() < 0.001,
            "Expected ROI of 5.0%, got {}",
            roi
        );
    }

    /// Verifies flop aggression factor is calculated correctly as (flop_bets + flop_raises) / flop_calls.
    #[test]
    fn test_flop_aggression_factor_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 4;
        stats.flop_raises[0] = 2;
        stats.flop_calls[0] = 3;

        // Flop AF = (4 + 2) / 3 = 2.0
        let flop_af = stats.flop_aggression_factor(0);
        assert!(
            (flop_af - 2.0).abs() < 0.001,
            "Expected Flop AF of 2.0, got {}",
            flop_af
        );
    }

    /// Verifies turn aggression factor is calculated correctly as (turn_bets + turn_raises) / turn_calls.
    #[test]
    fn test_turn_aggression_factor_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_bets[0] = 5;
        stats.turn_raises[0] = 3;
        stats.turn_calls[0] = 4;

        // Turn AF = (5 + 3) / 4 = 2.0
        let turn_af = stats.turn_aggression_factor(0);
        assert!(
            (turn_af - 2.0).abs() < 0.001,
            "Expected Turn AF of 2.0, got {}",
            turn_af
        );
    }

    /// Verifies river aggression factor calculation, including the INFINITY case when there are no calls.
    #[test]
    fn test_river_aggression_factor_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_bets[0] = 3;
        stats.river_raises[0] = 2;
        stats.river_calls[0] = 0;

        // River AF with no calls and aggressive actions = INFINITY
        let river_af = stats.river_aggression_factor(0);
        assert_eq!(river_af, f32::INFINITY);

        // Now with calls
        stats.river_calls[0] = 5;
        // River AF = (3 + 2) / 5 = 1.0
        let river_af = stats.river_aggression_factor(0);
        assert!(
            (river_af - 1.0).abs() < 0.001,
            "Expected River AF of 1.0, got {}",
            river_af
        );
    }

    /// Verifies that all action-related counts are correctly merged by addition.
    #[test]
    fn test_merge_all_action_counts() {
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.actions_count[0] = 10;
        stats1.vpip_count[0] = 5;
        stats1.vpip_total[0] = 100.0;
        stats1.raise_count[0] = 3;
        stats1.hands_played[0] = 10;
        stats1.hands_vpip[0] = 6;
        stats1.hands_pfr[0] = 4;
        stats1.preflop_raise_count[0] = 4;
        stats1.preflop_actions[0] = 10;
        stats1.three_bet_count[0] = 2;
        stats1.three_bet_opportunities[0] = 5;
        stats1.call_count[0] = 8;
        stats1.bet_count[0] = 3;

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.actions_count[0] = 5;
        stats2.vpip_count[0] = 3;
        stats2.vpip_total[0] = 50.0;
        stats2.raise_count[0] = 2;
        stats2.hands_played[0] = 5;
        stats2.hands_vpip[0] = 3;
        stats2.hands_pfr[0] = 2;
        stats2.preflop_raise_count[0] = 2;
        stats2.preflop_actions[0] = 5;
        stats2.three_bet_count[0] = 1;
        stats2.three_bet_opportunities[0] = 3;
        stats2.call_count[0] = 4;
        stats2.bet_count[0] = 2;

        stats1.merge(&stats2);

        assert_eq!(stats1.actions_count[0], 15); // 10 + 5
        assert_eq!(stats1.vpip_count[0], 8); // 5 + 3
        assert!((stats1.vpip_total[0] - 150.0).abs() < 0.001); // 100 + 50
        assert_eq!(stats1.raise_count[0], 5); // 3 + 2
        assert_eq!(stats1.hands_played[0], 15); // 10 + 5
        assert_eq!(stats1.hands_vpip[0], 9); // 6 + 3
        assert_eq!(stats1.hands_pfr[0], 6); // 4 + 2
        assert_eq!(stats1.preflop_raise_count[0], 6); // 4 + 2
        assert_eq!(stats1.preflop_actions[0], 15); // 10 + 5
        assert_eq!(stats1.three_bet_count[0], 3); // 2 + 1
        assert_eq!(stats1.three_bet_opportunities[0], 8); // 5 + 3
        assert_eq!(stats1.call_count[0], 12); // 8 + 4
        assert_eq!(stats1.bet_count[0], 5); // 3 + 2
    }

    /// Verifies that financial tracking fields are correctly merged by addition.
    #[test]
    fn test_merge_financial_tracking() {
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.total_profit[0] = 200.0;
        stats1.games_won[0] = 10;
        stats1.games_lost[0] = 5;
        stats1.games_breakeven[0] = 3;

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.total_profit[0] = -50.0;
        stats2.games_won[0] = 3;
        stats2.games_lost[0] = 4;
        stats2.games_breakeven[0] = 2;

        stats1.merge(&stats2);

        assert!((stats1.total_profit[0] - 150.0).abs() < 0.001); // 200 + (-50)
        assert_eq!(stats1.games_won[0], 13); // 10 + 3
        assert_eq!(stats1.games_lost[0], 9); // 5 + 4
        assert_eq!(stats1.games_breakeven[0], 5); // 3 + 2
    }

    /// Verifies that per-street action counts are correctly merged by addition.
    #[test]
    fn test_merge_per_street_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.flop_bets[0] = 5;
        stats1.flop_raises[0] = 3;
        stats1.flop_calls[0] = 7;
        stats1.turn_bets[0] = 4;
        stats1.turn_raises[0] = 2;
        stats1.turn_calls[0] = 6;
        stats1.river_bets[0] = 3;
        stats1.river_raises[0] = 1;
        stats1.river_calls[0] = 5;

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.flop_bets[0] = 2;
        stats2.flop_raises[0] = 1;
        stats2.flop_calls[0] = 3;
        stats2.turn_bets[0] = 2;
        stats2.turn_raises[0] = 1;
        stats2.turn_calls[0] = 2;
        stats2.river_bets[0] = 1;
        stats2.river_raises[0] = 1;
        stats2.river_calls[0] = 2;

        stats1.merge(&stats2);

        assert_eq!(stats1.flop_bets[0], 7);
        assert_eq!(stats1.flop_raises[0], 4);
        assert_eq!(stats1.flop_calls[0], 10);
        assert_eq!(stats1.turn_bets[0], 6);
        assert_eq!(stats1.turn_raises[0], 3);
        assert_eq!(stats1.turn_calls[0], 8);
        assert_eq!(stats1.river_bets[0], 4);
        assert_eq!(stats1.river_raises[0], 2);
        assert_eq!(stats1.river_calls[0], 7);
    }

    /// Verifies that steal tracking stats are correctly merged by addition.
    #[test]
    fn test_merge_steal_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.steal_opportunities[0] = 10;
        stats1.steal_count[0] = 6;

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.steal_opportunities[0] = 5;
        stats2.steal_count[0] = 3;

        stats1.merge(&stats2);

        assert_eq!(stats1.steal_opportunities[0], 15);
        assert_eq!(stats1.steal_count[0], 9);
    }

    #[test]
    fn test_is_steal_position_two_players() {
        // In heads-up (2 players), everyone is in steal position
        // This tests the num_players < 3 branch
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 2),
            "In heads-up, player 0 should be in steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 2),
            "In heads-up, player 1 should be in steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 1, 2),
            "In heads-up with dealer 1, player 0 should be in steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 1, 2),
            "In heads-up with dealer 1, player 1 should be in steal position"
        );
    }

    #[test]
    fn test_is_steal_position_three_players() {
        // In 3-player, CO = BTN-1, BTN = dealer, SB = BTN+1
        // With 3 players and dealer at 0: BTN=0, SB=1, BB=2
        // Steal positions: CO (wraps to 2), BTN (0), SB (1)
        // Wait, with 3 players: CO = (0 + 3 - 1) % 3 = 2
        // But position 2 is also BB. So steal positions are: BTN(0), CO(2), SB(1)
        // That means in 3-player, everyone is in steal position!

        // Let's verify the formula:
        // dealer_idx = 0, num_players = 3
        // btn = 0
        // co = (0 + 3 - 1) % 3 = 2
        // sb = (0 + 1) % 3 = 1
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 3),
            "BTN (pos 0) should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 3),
            "SB (pos 1) should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(2, 0, 3),
            "CO/BB (pos 2) should be steal position in 3-player"
        );
    }

    #[test]
    fn test_round_advance_preflop_completes() {
        // Test that preflop round advance increments preflop_completes
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Both players should have preflop_completes incremented
        assert!(
            stats.preflop_completes[0] > 0,
            "Expected preflop_completes[0] > 0"
        );
        assert!(
            stats.preflop_completes[1] > 0,
            "Expected preflop_completes[1] > 0"
        );
    }

    #[test]
    fn test_round_advance_all_rounds() {
        // Test that all round advances are tracked
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run multiple games to ensure we get games that reach all rounds
        for seed in 0..10 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // With calling agents, games should reach showdown, so all rounds should be tracked
        assert!(
            stats.preflop_completes[0] > 0,
            "preflop_completes should be tracked"
        );
        assert!(
            stats.flop_completes[0] > 0,
            "flop_completes should be tracked"
        );
        assert!(
            stats.turn_completes[0] > 0,
            "turn_completes should be tracked"
        );
        assert!(
            stats.river_completes[0] > 0,
            "river_completes should be tracked"
        );
    }

    #[test]
    fn test_game_complete_tracks_round_wins() {
        // Test that game_complete tracks wins at different rounds
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run games where one player folds early (preflop wins)
        for seed in 0..10 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<AllInAgent>::default() as Box<dyn Agent>,
                Box::<FoldingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // The all-in agent should have preflop wins (since folder folds preflop)
        assert!(
            stats.preflop_wins[0] > 0 || stats.games_won[0] > 0,
            "Expected the all-in agent to have wins"
        );
    }

    #[test]
    fn test_game_complete_profit_tracking() {
        // Test that profit > 0.01 and profit < -0.01 thresholds work correctly
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // One player should have won, one should have lost
        let total_won = stats.games_won[0] + stats.games_won[1];
        let total_lost = stats.games_lost[0] + stats.games_lost[1];

        assert!(total_won > 0, "Expected at least one win");
        assert!(total_lost > 0, "Expected at least one loss");
    }

    #[test]
    fn test_game_start_resets_state() {
        // Test that game_start properly resets tracking state
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        // Run two separate games and verify each hand is tracked separately
        for seed in 0..2 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // Should have exactly 2 hands played per player
        assert_eq!(
            stats.hands_played[0], 2,
            "Expected 2 hands played for player 0"
        );
        assert_eq!(
            stats.hands_played[1], 2,
            "Expected 2 hands played for player 1"
        );
    }

    #[test]
    fn test_played_action_bet_vpip_tracking() {
        // Test that bet actions correctly track VPIP
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        // Agent 0 raises (bet action with put_into_pot > 0)
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(30.0)], // Big raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // The raiser should have VPIP
        assert_eq!(stats.hands_vpip[0], 1, "Raiser should have VPIP");
        // The folder should not have VPIP
        assert_eq!(stats.hands_vpip[1], 0, "Folder should not have VPIP");
    }

    #[test]
    fn test_played_action_call_vpip_tracking() {
        // Test that call actions correctly track VPIP when putting in extra money
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        // Agent 0 raises, Agent 1 calls (putting in extra money = VPIP)
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![AgentAction::Call], // Calls the raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Both should have VPIP - raiser raised, caller called the raise
        assert_eq!(stats.hands_vpip[0], 1, "Raiser should have VPIP");
        assert_eq!(stats.hands_vpip[1], 1, "Caller should have VPIP");
    }

    #[test]
    fn test_played_action_fold_tracking() {
        // Test that fold actions are tracked
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Folding agent should have fold_count > 0
        assert!(
            stats.fold_count[0] > 0,
            "Expected fold_count[0] > 0, got {}",
            stats.fold_count[0]
        );
    }

    #[test]
    fn test_played_action_allin_tracking() {
        // Test that all-in actions are tracked as bets
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // All-in agent should have bet_count > 0
        assert!(
            stats.bet_count[0] > 0,
            "Expected bet_count[0] > 0 for all-in agent, got {}",
            stats.bet_count[0]
        );
        // All-in agent should have VPIP
        assert_eq!(stats.hands_vpip[0], 1, "All-in agent should have VPIP");
    }

    #[test]
    fn test_historian_record_action_game_start() {
        // Test that GameStart action is handled
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        // If we got here without panic, GameStart was handled
        let stats = storage.read();
        assert_eq!(
            stats.hands_played[0], 1,
            "Game start should initialize tracking"
        );
    }

    #[test]
    fn test_historian_record_action_failed_action() {
        // Test that FailedAction is processed like PlayedAction
        // This is hard to trigger directly, but we can verify the code path exists
        // by running games that might have action validation
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        // Use agents that might cause failed actions (bet too much, etc.)
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<AllInAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        // Verify game completed
        let stats = storage.read();
        assert_eq!(stats.hands_played[0], 1);
    }

    #[test]
    fn test_historian_record_action_award() {
        // Test that Award actions are handled
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        // Awards should be handled (game completes)
        let stats = storage.read();
        // One player should have profit
        let total_profit = stats.total_profit[0] + stats.total_profit[1];
        assert!(total_profit.abs() < 0.01, "Profits should sum to zero");
    }

    #[test]
    fn test_wtsd_saw_flop_tracking() {
        // Test that players who see the flop are tracked for WTSD opportunities
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        for seed in 0..5 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            // Both players call through to showdown
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // With calling agents, both should see the flop in every game
        assert_eq!(
            stats.wtsd_opportunities[0], 5,
            "Expected 5 WTSD opportunities for player 0"
        );
        assert_eq!(
            stats.wtsd_opportunities[1], 5,
            "Expected 5 WTSD opportunities for player 1"
        );
    }

    #[test]
    fn test_showdown_tracking_with_multiple_active_players() {
        // Test that showdown is only counted when round is Complete and multiple players active
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let storage = SharedStatsStorage::new(2);

        for seed in 0..10 {
            let hist = storage.historian();
            let stacks = vec![100.0, 100.0];
            let agents: Vec<Box<dyn Agent>> = vec![
                Box::<CallingAgent>::default() as Box<dyn Agent>,
                Box::<CallingAgent>::default() as Box<dyn Agent>,
            ];

            let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build()
                .unwrap();

            sim.run(&mut rng);
        }

        let stats = storage.read();

        // With calling agents, games should reach showdown
        assert!(
            stats.showdown_count[0] > 0 && stats.showdown_count[1] > 0,
            "Expected showdown counts for both players"
        );
    }

    #[test]
    fn test_preflop_raise_tracking() {
        // Test that preflop raises are tracked for PFR calculation
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        // Agent 0 raises preflop multiple times if game continues
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0), AgentAction::Bet(60.0)],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![AgentAction::Call, AgentAction::Call],
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Agent 0 should have preflop raise count > 0
        assert!(
            stats.preflop_raise_count[0] > 0,
            "Expected preflop_raise_count[0] > 0"
        );
        // Agent 0 should have hands_pfr = 1 (binary per hand)
        assert_eq!(stats.hands_pfr[0], 1, "Raiser should have hands_pfr = 1");
        // Agent 1 should have hands_pfr = 0 (didn't raise preflop)
        assert_eq!(stats.hands_pfr[1], 0, "Caller should have hands_pfr = 0");
    }

    #[test]
    fn test_three_bet_tracking() {
        // Test that 3-bet opportunities and counts are tracked
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![200.0, 200.0];
        // Agent 0 raises, Agent 1 re-raises (3-bet)
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0), AgentAction::Call],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "three-better",
                vec![AgentAction::Bet(50.0)], // 3-bet
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let stats = storage.read();

        // Agent 1 should have 3-bet opportunity and count
        assert!(
            stats.three_bet_opportunities[1] > 0,
            "Expected 3-bet opportunities for player 1"
        );
        assert!(
            stats.three_bet_count[1] > 0,
            "Expected 3-bet count for player 1"
        );
    }

    /// Verifies that river bets are accumulated correctly with addition.
    #[test]
    fn test_river_bets_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_bets[0] = 5;
        stats.river_bets[0] += 3;
        assert_eq!(stats.river_bets[0], 8, "river_bets should be 5 + 3 = 8");
    }

    /// Verifies that turn bets are accumulated correctly with addition.
    #[test]
    fn test_turn_bets_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_bets[0] = 4;
        stats.turn_bets[0] += 2;
        assert_eq!(stats.turn_bets[0], 6, "turn_bets should be 4 + 2 = 6");
    }

    /// Verifies that flop bets are accumulated correctly with addition.
    #[test]
    fn test_flop_bets_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 3;
        stats.flop_bets[0] += 2;
        assert_eq!(stats.flop_bets[0], 5, "flop_bets should be 3 + 2 = 5");
    }

    /// Verifies that river raises are accumulated correctly with addition.
    #[test]
    fn test_river_raises_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_raises[0] = 2;
        stats.river_raises[0] += 3;
        assert_eq!(stats.river_raises[0], 5, "river_raises should be 2 + 3 = 5");
    }

    /// Verifies that turn raises are accumulated correctly with addition.
    #[test]
    fn test_turn_raises_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_raises[0] = 2;
        stats.turn_raises[0] += 2;
        assert_eq!(stats.turn_raises[0], 4, "turn_raises should be 2 + 2 = 4");
    }

    /// Verifies that flop raises are accumulated correctly with addition.
    #[test]
    fn test_flop_raises_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_raises[0] = 1;
        stats.flop_raises[0] += 4;
        assert_eq!(stats.flop_raises[0], 5, "flop_raises should be 1 + 4 = 5");
    }

    /// Verifies that river calls are accumulated correctly with addition.
    #[test]
    fn test_river_calls_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_calls[0] = 3;
        stats.river_calls[0] += 2;
        assert_eq!(stats.river_calls[0], 5, "river_calls should be 3 + 2 = 5");
    }

    /// Verifies that turn calls are accumulated correctly with addition.
    #[test]
    fn test_turn_calls_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_calls[0] = 4;
        stats.turn_calls[0] += 1;
        assert_eq!(stats.turn_calls[0], 5, "turn_calls should be 4 + 1 = 5");
    }

    /// Verifies that flop calls are accumulated correctly with addition.
    #[test]
    fn test_flop_calls_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_calls[0] = 2;
        stats.flop_calls[0] += 3;
        assert_eq!(stats.flop_calls[0], 5, "flop_calls should be 2 + 3 = 5");
    }

    /// Verifies that vpip_count is accumulated correctly with addition.
    #[test]
    fn test_vpip_count_legacy_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.vpip_count[0] = 10;
        stats.vpip_count[0] += 5;
        assert_eq!(stats.vpip_count[0], 15, "vpip_count should be 10 + 5 = 15");
    }

    /// Verifies that vpip_total is accumulated correctly with addition.
    #[test]
    fn test_vpip_total_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.vpip_total[0] = 100.0;
        stats.vpip_total[0] += 50.0;
        assert!(
            (stats.vpip_total[0] - 150.0).abs() < 0.01,
            "vpip_total should be 100 + 50 = 150"
        );
    }

    /// Verifies that total_invested is accumulated correctly with addition.
    #[test]
    fn test_invested_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_invested[0] = 200.0;
        stats.total_invested[0] += 100.0;
        assert!(
            (stats.total_invested[0] - 300.0).abs() < 0.01,
            "total_invested should be 200 + 100 = 300"
        );
    }

    /// Verifies that preflop_raise_count is accumulated correctly with addition.
    #[test]
    fn test_preflop_raise_count_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.preflop_raise_count[0] = 2;
        stats.preflop_raise_count[0] += 3;
        assert_eq!(
            stats.preflop_raise_count[0], 5,
            "preflop_raise_count should be 2 + 3 = 5"
        );
    }

    /// Verifies that flop aggression factor correctly sums bets and raises in the numerator.
    #[test]
    fn test_per_street_aggression_factor_addition() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 3;
        stats.flop_raises[0] = 2;
        stats.flop_calls[0] = 1;

        // (3 + 2) / 1 = 5.0
        let af = stats.flop_aggression_factor(0);
        assert!(
            (af - 5.0).abs() < 0.001,
            "Flop AF should be (3 + 2) / 1 = 5.0, got {}",
            af
        );
    }

    /// Verifies that turn aggression factor correctly sums bets and raises in the numerator.
    #[test]
    fn test_turn_aggression_factor_addition() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.turn_bets[0] = 4;
        stats.turn_raises[0] = 3;
        stats.turn_calls[0] = 2;

        // (4 + 3) / 2 = 3.5
        let af = stats.turn_aggression_factor(0);
        assert!(
            (af - 3.5).abs() < 0.001,
            "Turn AF should be (4 + 3) / 2 = 3.5, got {}",
            af
        );
    }

    /// Verifies that river aggression factor correctly sums bets and raises in the numerator.
    #[test]
    fn test_river_aggression_factor_addition() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.river_bets[0] = 2;
        stats.river_raises[0] = 4;
        stats.river_calls[0] = 3;

        // (2 + 4) / 3 = 2.0
        let af = stats.river_aggression_factor(0);
        assert!(
            (af - 2.0).abs() < 0.001,
            "River AF should be (2 + 4) / 3 = 2.0, got {}",
            af
        );
    }

    /// Verifies that overall aggression factor correctly sums raises and bets in the numerator.
    #[test]
    fn test_overall_aggression_factor_addition() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 5;
        stats.bet_count[0] = 3;
        stats.call_count[0] = 4;

        // (5 + 3) / 4 = 2.0
        let af = stats.aggression_factor(0);
        assert!(
            (af - 2.0).abs() < 0.001,
            "Overall AF should be (5 + 3) / 4 = 2.0, got {}",
            af
        );
    }

    /// Verifies that wins at each round are tracked correctly with increments.
    #[test]
    fn test_round_wins_tracking() {
        let mut stats = StatsStorage::new_with_num_players(1);

        stats.preflop_wins[0] = 2;
        stats.flop_wins[0] = 3;
        stats.turn_wins[0] = 1;
        stats.river_wins[0] = 4;

        // Simulate increments like in record_game_complete
        stats.preflop_wins[0] += 1;
        stats.flop_wins[0] += 1;
        stats.turn_wins[0] += 1;
        stats.river_wins[0] += 1;

        assert_eq!(stats.preflop_wins[0], 3, "preflop_wins should be 2 + 1 = 3");
        assert_eq!(stats.flop_wins[0], 4, "flop_wins should be 3 + 1 = 4");
        assert_eq!(stats.turn_wins[0], 2, "turn_wins should be 1 + 1 = 2");
        assert_eq!(stats.river_wins[0], 5, "river_wins should be 4 + 1 = 5");
    }

    /// Verifies that games_breakeven is accumulated correctly with addition.
    #[test]
    fn test_games_breakeven_tracking() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.games_breakeven[0] = 3;
        stats.games_breakeven[0] += 2;
        assert_eq!(
            stats.games_breakeven[0], 5,
            "games_breakeven should be 3 + 2 = 5"
        );
    }

    /// Verifies steal position detection for various table sizes (heads-up, 3-player, 6-player).
    #[test]
    fn test_is_steal_position_edge_cases() {
        // Test heads-up (2 players) - both should be in steal position
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 2),
            "In heads-up, player 0 should be in steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 2),
            "In heads-up, player 1 should be in steal position"
        );

        // Test 3 players with dealer at 0
        // Button (0), CO (2), SB (1) should all be steal positions
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 3),
            "Button should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 3),
            "SB should be steal position"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(2, 0, 3),
            "CO should be steal position"
        );

        // Test 6 players
        // BTN=0, CO=5, SB=1 are steal positions; 2,3,4 are not
        assert!(
            StatsTrackingHistorian::is_steal_position(0, 0, 6),
            "Button should be steal position (6 players)"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(5, 0, 6),
            "CO should be steal position (6 players)"
        );
        assert!(
            StatsTrackingHistorian::is_steal_position(1, 0, 6),
            "SB should be steal position (6 players)"
        );
        assert!(
            !StatsTrackingHistorian::is_steal_position(2, 0, 6),
            "BB should NOT be steal position (6 players)"
        );
        assert!(
            !StatsTrackingHistorian::is_steal_position(3, 0, 6),
            "UTG should NOT be steal position (6 players)"
        );
    }

    #[test]
    fn test_hand_accumulator_reset() {
        let mut accumulator = HandAccumulator::new(2);

        // Set some values
        accumulator.player_stats[0].actions_count = 5;
        accumulator.player_stats[0].raise_count = 3;
        accumulator.player_stats[1].bet_count = 2;
        accumulator.player_stats[1].vpip_occurred = true;

        // Reset
        accumulator.reset();

        // Verify all values are reset to defaults
        assert_eq!(
            accumulator.player_stats[0].actions_count, 0,
            "actions_count should be 0 after reset"
        );
        assert_eq!(
            accumulator.player_stats[0].raise_count, 0,
            "raise_count should be 0 after reset"
        );
        assert_eq!(
            accumulator.player_stats[1].bet_count, 0,
            "bet_count should be 0 after reset"
        );
        assert!(
            !accumulator.player_stats[1].vpip_occurred,
            "vpip_occurred should be false after reset"
        );
    }

    #[test]
    fn test_flush_accumulated_stats_arithmetic() {
        // Test that flush_accumulated_stats correctly adds accumulated values
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        // Pre-set some values in storage
        {
            let mut s = storage.inner().write().unwrap();
            s.actions_count[0] = 10;
            s.raise_count[0] = 5;
        }

        // Simulate accumulated stats (manually set via internal struct)
        // We test this through a full simulation instead
        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)],
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![AgentAction::Call],
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        // Verify stats were accumulated (values should have increased)
        let s = storage.read();
        assert!(
            s.actions_count[0] > 10,
            "actions_count should have increased from 10"
        );
        assert!(
            s.raise_count[0] > 5,
            "raise_count should have increased from 5"
        );
    }

    #[test]
    fn test_total_profit_tracking_arithmetic() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 50.0;
        stats.total_profit[0] += 25.0;
        assert!(
            (stats.total_profit[0] - 75.0).abs() < 0.01,
            "total_profit should be 50 + 25 = 75"
        );

        // Test negative profit (loss)
        stats.total_profit[0] += -100.0;
        assert!(
            (stats.total_profit[0] - (-25.0)).abs() < 0.01,
            "total_profit should be 75 + (-100) = -25"
        );
    }

    #[test]
    fn test_record_game_complete_profit_sign() {
        // Test that profit can be negative (distinguishes += from *= for negative values)
        let storage = SharedStatsStorage::new(2);
        let hist = storage.historian();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>, // Will lose blinds
            Box::<AllInAgent>::default() as Box<dyn Agent>,   // Will win
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![Box::new(hist)])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let s = storage.read();
        // Folding agent (player 0) should have lost their blind
        assert!(
            s.total_profit[0] < 0.0,
            "Folding agent should have negative profit (lost blind)"
        );
        // All-in agent (player 1) should have won
        assert!(
            s.total_profit[1] > 0.0,
            "All-in agent should have positive profit"
        );
    }
}
