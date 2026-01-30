/// Module that can generate possible cards for a starting hand.
mod starting_hand;
/// Export `StartingHand`
pub use self::starting_hand::{StartingHand, Suitedness};

/// Module for `MonteCarloGame` that holds the current state of the deck for
/// simulation.
mod monte_carlo_game;
/// Export `MonteCarloGame`
pub use self::monte_carlo_game::MonteCarloGame;

/// Module with all the starting hand parsing code.
mod parse;
/// Export `RangeParser`
pub use self::parse::RangeParser;

/// Module for calculating outs and equity by enumerating all possible board
/// completions.
mod outs_calculator;
/// Export `OutsCalculator` and `PlayerOuts`
pub use self::outs_calculator::{OutsCalculator, PlayerOutcome};

/// Module for pre-flop charts representing action frequencies.
mod preflop_chart;
/// Export pre-flop chart types
pub use self::preflop_chart::{PreflopActionType, PreflopChart, PreflopHand, PreflopStrategy};
