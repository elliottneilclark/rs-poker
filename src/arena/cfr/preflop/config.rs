//! Configuration types for preflop chart-based CFR agents.
//!
//! This module provides configuration structures that allow CFR agents to use
//! pre-computed preflop charts instead of exploring preflop decisions.

use crate::holdem::PreflopChart;
use serde::{Deserialize, Serialize};

fn default_raise_size_bb() -> f32 {
    2.5
}

fn default_three_bet_multiplier() -> f32 {
    3.0
}

/// Configuration for preflop chart-based play.
///
/// This configuration specifies which hands to play from each position
/// and how to size bets during preflop.
///
/// # Example JSON
///
/// ```json
/// {
///   "raise_size_bb": 2.5,
///   "three_bet_multiplier": 3.0,
///   "charts": [
///     { "AA": {"Raise": 1.0}, "KK": {"Raise": 1.0} },
///     { "AA": {"Raise": 1.0} }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopChartConfig {
    /// Charts indexed by position (distance from button).
    /// Index 0 = Button, 1 = Small Blind, 2 = Big Blind, 3+ = early positions.
    ///
    /// If a position index exceeds the available charts, the last chart is used.
    /// This allows specifying fewer charts (e.g., just one "tight" chart for all positions)
    /// while still having position-specific play when desired.
    pub charts: Vec<PreflopChart>,

    /// Default raise size as multiple of big blind for open raises.
    /// Standard is 2.5bb from most positions.
    #[serde(default = "default_raise_size_bb")]
    pub raise_size_bb: f32,

    /// Multiplier for 3-bet sizing (3-bet = this * opponent's raise).
    /// Standard is 3x the opponent's raise from in position.
    #[serde(default = "default_three_bet_multiplier")]
    pub three_bet_multiplier: f32,
}

impl Default for PreflopChartConfig {
    fn default() -> Self {
        Self {
            charts: vec![PreflopChart::new()],
            raise_size_bb: default_raise_size_bb(),
            three_bet_multiplier: default_three_bet_multiplier(),
        }
    }
}

impl PreflopChartConfig {
    /// Create a new config with the given charts.
    pub fn new(charts: Vec<PreflopChart>) -> Self {
        Self {
            charts,
            ..Default::default()
        }
    }

    /// Create a config with a single chart used for all positions.
    pub fn with_single_chart(chart: PreflopChart) -> Self {
        Self {
            charts: vec![chart],
            ..Default::default()
        }
    }

    /// Get the chart for a given position relative to button.
    ///
    /// Position 0 = Button, 1 = Small Blind, 2 = Big Blind, etc.
    /// If position exceeds available charts, returns the last chart.
    pub fn chart_for_position(&self, position: usize) -> &PreflopChart {
        if self.charts.is_empty() {
            // This shouldn't happen with Default, but handle gracefully
            panic!("PreflopChartConfig has no charts");
        }
        let idx = position.min(self.charts.len() - 1);
        &self.charts[idx]
    }

    /// Calculate the position relative to the big blind.
    ///
    /// Returns the distance from the big blind position (counter-clockwise):
    /// - 0 = Big Blind
    /// - 1 = Small Blind (or Button in heads-up)
    /// - 2 = Button (for 3+ players)
    /// - 3 = Cutoff
    /// - 4 = Hijack
    /// - 5+ = Earlier positions (UTG, etc.)
    ///
    /// This ordering is designed so that if fewer charts are provided than
    /// positions, the "fallback" (last chart) represents the tightest/earliest
    /// position range, which is appropriate for unspecified early positions.
    ///
    /// Note: In heads-up (2 players), the button posts the small blind and
    /// acts first preflop. The BB is 1 position after the dealer, not 2.
    pub fn calculate_position(player_idx: usize, dealer_idx: usize, num_players: usize) -> usize {
        // In heads-up, BB is 1 seat after dealer (BTN posts SB)
        // In 3+ players, BB is 2 seats after dealer
        let bb_offset = if num_players == 2 { 1 } else { 2 };
        let bb_idx = (dealer_idx + bb_offset) % num_players;
        // Distance from BB (counter-clockwise = positions acting before BB)
        (bb_idx + num_players - player_idx) % num_players
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.charts.is_empty() {
            return Err("At least one preflop chart is required".to_string());
        }
        if self.raise_size_bb <= 0.0 {
            return Err("raise_size_bb must be positive".to_string());
        }
        if self.three_bet_multiplier <= 0.0 {
            return Err("three_bet_multiplier must be positive".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use crate::holdem::{PreflopActionType, PreflopHand, PreflopStrategy};

    #[test]
    fn test_default_config() {
        let config = PreflopChartConfig::default();
        assert_eq!(config.charts.len(), 1);
        assert_eq!(config.raise_size_bb, 2.5);
        assert_eq!(config.three_bet_multiplier, 3.0);
    }

    #[test]
    fn test_position_calculation() {
        // 6-player table, dealer at position 3
        // Positions relative to BB (position 0):
        // Player 5 = BB (0)
        // Player 4 = SB (1)
        // Player 3 = BTN (2)
        // Player 2 = CO (3)
        // Player 1 = HJ (4)
        // Player 0 = UTG (5)

        let num_players = 6;
        let dealer_idx = 3;

        assert_eq!(
            PreflopChartConfig::calculate_position(5, dealer_idx, num_players),
            0
        ); // BB
        assert_eq!(
            PreflopChartConfig::calculate_position(4, dealer_idx, num_players),
            1
        ); // SB
        assert_eq!(
            PreflopChartConfig::calculate_position(3, dealer_idx, num_players),
            2
        ); // BTN
        assert_eq!(
            PreflopChartConfig::calculate_position(2, dealer_idx, num_players),
            3
        ); // CO
        assert_eq!(
            PreflopChartConfig::calculate_position(1, dealer_idx, num_players),
            4
        ); // HJ
        assert_eq!(
            PreflopChartConfig::calculate_position(0, dealer_idx, num_players),
            5
        ); // UTG
    }

    #[test]
    fn test_chart_for_position_single_chart() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig::with_single_chart(chart);

        // All positions should use the same chart
        for pos in 0..10 {
            let c = config.chart_for_position(pos);
            assert!(c.get(&aa).is_some());
        }
    }

    #[test]
    fn test_chart_for_position_multiple_charts() {
        let mut btn_chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        btn_chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let mut utg_chart = PreflopChart::new();
        // UTG chart only has AA, nothing else
        utg_chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        let kk = PreflopHand::new(Value::King, Value::King, false);
        utg_chart.set(kk, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig::new(vec![btn_chart, utg_chart.clone()]);

        // Position 0 (BTN) uses first chart
        let btn = config.chart_for_position(0);
        assert!(btn.get(&aa).is_some());
        assert!(btn.get(&kk).is_none()); // BTN chart doesn't have KK

        // Position 1+ uses second chart (which has KK)
        let other = config.chart_for_position(1);
        assert!(other.get(&kk).is_some());
    }

    #[test]
    fn test_validate_empty_charts() {
        let config = PreflopChartConfig {
            charts: vec![],
            raise_size_bb: 2.5,
            three_bet_multiplier: 3.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_raise_size() {
        let config = PreflopChartConfig {
            charts: vec![PreflopChart::new()],
            raise_size_bb: 0.0,
            three_bet_multiplier: 3.0,
        };
        assert!(config.validate().is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig {
            charts: vec![chart],
            raise_size_bb: 3.0,
            three_bet_multiplier: 3.5,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: PreflopChartConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.raise_size_bb, 3.0);
        assert_eq!(parsed.three_bet_multiplier, 3.5);
        assert_eq!(parsed.charts.len(), 1);
    }
}
