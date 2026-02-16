use core::fmt;
use std::fmt::Display;

use approx::abs_diff_eq;
use rand::rng;
use thiserror::Error;

use crate::core::{Card, CardBitSet, Hand, PlayerBitSet};

use super::errors::GameStateError;

/// Maximum number of players supported (based on PlayerBitSet using u16).
pub const MAX_PLAYERS: usize = 16;

/// Errors that can occur when building a GameState.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GameStateBuilderError {
    #[error("stacks are required")]
    MissingStacks,

    #[error("big_blind is required")]
    MissingBigBlind,

    #[error("num_players must be between 2 and {max}, got {actual}", max = MAX_PLAYERS)]
    InvalidPlayerCount { actual: usize },

    #[error("dealer_idx {dealer_idx} must be less than num_players {num_players}")]
    InvalidDealerIndex {
        dealer_idx: usize,
        num_players: usize,
    },

    #[error("big_blind must be positive, got {0}")]
    InvalidBigBlind(f32),

    #[error("small_blind must be non-negative, got {0}")]
    InvalidSmallBlind(f32),

    #[error("ante must be non-negative, got {0}")]
    InvalidAnte(f32),

    #[error("stack at index {index} must be non-negative, got {value}")]
    InvalidStack { index: usize, value: f32 },

    #[error("at least 2 players must have positive stacks")]
    InsufficientActivePlayers,

    #[error("hands length {hands_len} must equal num_players {num_players}")]
    HandsLengthMismatch {
        hands_len: usize,
        num_players: usize,
    },

    #[error("player_bet length {bet_len} must equal num_players {num_players}")]
    PlayerBetLengthMismatch { bet_len: usize, num_players: usize },

    #[error("board must have 0, 3, 4, or 5 cards, got {0}")]
    InvalidBoardSize(usize),

    #[error("duplicate card found: {0}")]
    DuplicateCard(Card),
}

/// Builder for constructing `GameState` with validation.
///
/// # Example
///
/// ```
/// use rs_poker::arena::GameStateBuilder;
///
/// let game_state = GameStateBuilder::new()
///     .stacks(vec![100.0, 100.0])
///     .big_blind(10.0)
///     .build()
///     .unwrap();
///
/// assert_eq!(game_state.num_players, 2);
/// assert_eq!(game_state.big_blind, 10.0);
/// assert_eq!(game_state.small_blind, 5.0); // defaults to big_blind / 2
/// ```
#[derive(Default, Clone)]
pub struct GameStateBuilder {
    // Required (no defaults)
    stacks: Option<Vec<f32>>,
    big_blind: Option<f32>,

    // Optional with defaults
    small_blind: Option<f32>,                 // Default: big_blind / 2
    ante: Option<f32>,                        // Default: 0.0
    dealer_idx: Option<usize>,                // Default: 0
    max_raises_per_round: Option<Option<u8>>, // Default: Some(3)

    // For mid-game states (defaults for new games)
    round: Option<Round>,          // Default: Round::Starting
    board: Option<Vec<Card>>,      // Default: vec![]
    hands: Option<Vec<Hand>>,      // Default: vec![Hand::default(); n]
    player_bet: Option<Vec<f32>>,  // Default: vec![0.0; n]
    round_data: Option<RoundData>, // Default: computed
}

impl GameStateBuilder {
    /// Create a new `GameStateBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the stack sizes for each player. Required.
    pub fn stacks(mut self, stacks: Vec<f32>) -> Self {
        self.stacks = Some(stacks);
        self
    }

    /// Set the big blind size. Required.
    pub fn big_blind(mut self, bb: f32) -> Self {
        self.big_blind = Some(bb);
        self
    }

    /// Set the small blind size. Defaults to `big_blind / 2`.
    pub fn small_blind(mut self, sb: f32) -> Self {
        self.small_blind = Some(sb);
        self
    }

    /// Set the ante size. Defaults to `0.0`.
    pub fn ante(mut self, ante: f32) -> Self {
        self.ante = Some(ante);
        self
    }

    /// Set the dealer index. Defaults to `0`.
    pub fn dealer_idx(mut self, idx: usize) -> Self {
        self.dealer_idx = Some(idx);
        self
    }

    /// Set the maximum raises per round. Defaults to `Some(3)`.
    /// Use `None` for unlimited raises.
    pub fn max_raises_per_round(mut self, max: Option<u8>) -> Self {
        self.max_raises_per_round = Some(max);
        self
    }

    /// Convenience method to create stacks with `n` players, each with `stack` chips.
    pub fn num_players_with_stack(mut self, n: usize, stack: f32) -> Self {
        self.stacks = Some(vec![stack; n]);
        self
    }

    /// Convenience method to set both big and small blinds at once.
    pub fn blinds(mut self, big: f32, small: f32) -> Self {
        self.big_blind = Some(big);
        self.small_blind = Some(small);
        self
    }

    /// Set the current round. Defaults to `Round::Starting`.
    pub fn round(mut self, round: Round) -> Self {
        self.round = Some(round);
        self
    }

    /// Set the board cards. Defaults to empty.
    pub fn board(mut self, board: Vec<Card>) -> Self {
        self.board = Some(board);
        self
    }

    /// Set the hands for each player.
    pub fn hands(mut self, hands: Vec<Hand>) -> Self {
        self.hands = Some(hands);
        self
    }

    /// Set the player bets.
    pub fn player_bet(mut self, bets: Vec<f32>) -> Self {
        self.player_bet = Some(bets);
        self
    }

    /// Set the round data directly.
    pub fn round_data(mut self, rd: RoundData) -> Self {
        self.round_data = Some(rd);
        self
    }

    /// Build the `GameState`, validating all inputs.
    pub fn build(self) -> Result<GameState, GameStateBuilderError> {
        // Check required fields
        let stacks = self.stacks.ok_or(GameStateBuilderError::MissingStacks)?;
        let big_blind = self
            .big_blind
            .ok_or(GameStateBuilderError::MissingBigBlind)?;

        let num_players = stacks.len();

        // Validate player count
        if !(2..=MAX_PLAYERS).contains(&num_players) {
            return Err(GameStateBuilderError::InvalidPlayerCount {
                actual: num_players,
            });
        }

        // Validate big blind
        if big_blind <= 0.0 || big_blind.is_nan() {
            return Err(GameStateBuilderError::InvalidBigBlind(big_blind));
        }

        // Set defaults and validate small blind
        let small_blind = self.small_blind.unwrap_or(big_blind / 2.0);
        if small_blind < 0.0 || small_blind.is_nan() {
            return Err(GameStateBuilderError::InvalidSmallBlind(small_blind));
        }

        // Validate ante
        let ante = self.ante.unwrap_or(0.0);
        if ante < 0.0 || ante.is_nan() {
            return Err(GameStateBuilderError::InvalidAnte(ante));
        }

        // Validate stacks
        let round = self.round.unwrap_or(Round::Starting);
        let player_bet_ref = self.player_bet.as_deref();
        let mut active_count = 0;
        for (index, &value) in stacks.iter().enumerate() {
            if value < 0.0 || value.is_nan() {
                return Err(GameStateBuilderError::InvalidStack { index, value });
            }
            // A player is "active" if they have chips OR if they're all-in
            // (0 stack but has bet in a non-Starting round)
            let bet = player_bet_ref
                .and_then(|bets| bets.get(index).copied())
                .unwrap_or(0.0);
            if value > 0.0 || (bet > 0.0 && round != Round::Starting) {
                active_count += 1;
            }
        }
        // Only enforce minimum active players for new games (Starting round)
        // Mid-game states or tournament continuations may have fewer active players
        if active_count < 2 && round == Round::Starting {
            return Err(GameStateBuilderError::InsufficientActivePlayers);
        }

        // Validate dealer index
        let dealer_idx = self.dealer_idx.unwrap_or(0);
        if dealer_idx >= num_players {
            return Err(GameStateBuilderError::InvalidDealerIndex {
                dealer_idx,
                num_players,
            });
        }

        // Validate hands length if provided
        if let Some(ref hands) = self.hands
            && hands.len() != num_players
        {
            return Err(GameStateBuilderError::HandsLengthMismatch {
                hands_len: hands.len(),
                num_players,
            });
        }

        // Validate player_bet length if provided
        if let Some(ref bets) = self.player_bet
            && bets.len() != num_players
        {
            return Err(GameStateBuilderError::PlayerBetLengthMismatch {
                bet_len: bets.len(),
                num_players,
            });
        }

        // Validate board size
        let board = self.board.unwrap_or_default();
        let board_len = board.len();
        if board_len != 0 && board_len != 3 && board_len != 4 && board_len != 5 {
            return Err(GameStateBuilderError::InvalidBoardSize(board_len));
        }

        // Check for duplicate cards within the board only
        // We don't check hands because they can legitimately contain board cards
        // (e.g., when using 7-card hands for ranking purposes in tests)
        let mut card_set = CardBitSet::new();
        for card in &board {
            if card_set.contains(*card) {
                return Err(GameStateBuilderError::DuplicateCard(*card));
            }
            card_set.insert(*card);
        }

        let hands = self
            .hands
            .unwrap_or_else(|| vec![Hand::default(); num_players]);

        // Build defaults (round was already computed during validation)
        let player_bet = self.player_bet.unwrap_or_else(|| vec![0.0; num_players]);
        let max_raises_per_round = self.max_raises_per_round.unwrap_or(Some(3));

        let round_data = self.round_data.unwrap_or_else(|| {
            RoundData::new(
                num_players,
                big_blind,
                PlayerBitSet::new(num_players),
                dealer_idx,
            )
        });

        // Compute player_active, player_all_in, and total_pot
        let mut player_active = PlayerBitSet::new(num_players);
        let mut player_all_in = PlayerBitSet::default();
        let mut total_pot = 0.0;

        for (idx, (stack, bet)) in stacks.iter().zip(player_bet.iter()).enumerate() {
            total_pot += *bet;

            if *stack <= 0.0 {
                if *bet > 0.0 && round != Round::Starting {
                    // Player is out of money but has bet - they're all in
                    player_all_in.enable(idx);
                } else {
                    // Player has no money and can't play - sitting out
                    player_active.disable(idx);
                }
            }
        }

        Ok(GameState {
            num_players,
            starting_stacks: stacks.clone(),
            stacks,
            big_blind,
            small_blind,
            ante,
            player_active,
            player_all_in,
            player_bet,
            player_winnings: vec![0.0; num_players],
            dealer_idx,
            total_pot,
            hands,
            round,
            round_before: round,
            round_data,
            board,
            bb_posted: round != Round::Starting,
            sb_posted: round != Round::Starting,
            max_raises_per_round,
        })
    }
}

/// The round of the game.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Round {
    #[default]
    Starting,
    Ante,

    DealPreflop,
    Preflop,

    DealFlop,
    Flop,

    DealTurn,
    Turn,

    DealRiver,
    River,

    Showdown,
    Complete,
}

impl Display for Round {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Round::Starting => write!(f, "Starting"),

            Round::Ante => write!(f, "Ante"),

            Round::DealPreflop => write!(f, "Deal Preflop"),
            Round::Preflop => write!(f, "Preflop"),

            Round::DealFlop => write!(f, "Deal Flop"),
            Round::Flop => write!(f, "Flop"),

            Round::DealTurn => write!(f, "Deal Turn"),
            Round::Turn => write!(f, "Turn"),

            Round::DealRiver => write!(f, "Deal River"),
            Round::River => write!(f, "River"),

            Round::Showdown => write!(f, "Showdown"),
            Round::Complete => write!(f, "Complete"),
        }
    }
}

impl Round {
    pub fn advance(&self) -> Self {
        match *self {
            Round::Starting => Round::Ante,
            Round::Ante => Round::DealPreflop,
            Round::DealPreflop => Round::Preflop,
            Round::Preflop => Round::DealFlop,
            Round::DealFlop => Round::Flop,
            Round::Flop => Round::DealTurn,
            Round::DealTurn => Round::Turn,
            Round::Turn => Round::DealRiver,
            Round::DealRiver => Round::River,
            Round::River => Round::Showdown,
            Round::Showdown => Round::Complete,

            Round::Complete => Round::Complete,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct RoundData {
    // Which players were active starting this round.
    pub starting_player_active: PlayerBitSet,
    pub needs_action: PlayerBitSet,
    // The minimum allowed raise.
    pub min_raise: f32,
    // The value to be called.
    pub bet: f32,
    // How much each player has put in so far.
    pub player_bet: Vec<f32>,
    // The number of times anyone has put in money
    pub total_bet_count: u8,
    // The number of times anyone has increased the bet non-forced.
    pub total_raise_count: u8,
    // The number of forced bets (blinds, antes, straddles).
    pub forced_bet_count: u8,
    // The index of the next player to act.
    pub to_act_idx: usize,
}

impl RoundData {
    pub fn new(num_players: usize, min_raise: f32, active: PlayerBitSet, to_act: usize) -> Self {
        RoundData {
            needs_action: active,
            starting_player_active: active,
            min_raise,
            bet: 0.0,
            player_bet: vec![0.0; num_players],
            total_bet_count: 0,
            total_raise_count: 0,
            forced_bet_count: 0,
            to_act_idx: to_act,
        }
    }

    /// Create a new round data with the given bets.
    /// This is useful for creating a new round data that represents
    /// a round that is halfway through. For example, if we're trying
    /// to simulate a choosing to call an all in on the river.
    ///
    /// # Arguments
    ///
    /// * `num_players` - The number of players in the game.
    /// * `min_raise` - The minimum raise allowed in the round.
    /// * `active` - The players that are active in the round.
    /// * `to_act` - The index of the player that is next to act.
    /// * `player_bet` - The amount each player has bet so far.
    ///
    /// # Returns
    ///
    /// A new round data with the given bets, the bets are
    /// used to assume other values of the round.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_poker::arena::game_state::RoundData;
    /// use rs_poker::core::PlayerBitSet;
    ///
    /// let num_players = 3;
    /// let min_raise = 10.0;
    /// let active = PlayerBitSet::new(num_players);
    ///
    /// let player_bet = vec![0.0, 10.0, 20.0];
    /// let to_act = 0;
    ///
    /// let round_data = RoundData::new_with_bets(min_raise, active, to_act, player_bet);
    ///
    /// assert_eq!(round_data.bet, 20.0);
    ///
    /// assert_eq!(round_data.total_bet_count, 2);
    ///
    /// assert_eq!(round_data.total_raise_count, 2);
    /// ```
    pub fn new_with_bets(
        min_raise: f32,
        active: PlayerBitSet,
        to_act: usize,
        player_bet: Vec<f32>,
    ) -> Self {
        let bet: f32 = player_bet.iter().fold(0.0, |acc, &x| acc.max(x));

        let total_raise_count = player_bet.iter().filter(|&&x| x > 0.0).count() as u8;

        RoundData {
            needs_action: active,
            starting_player_active: active,
            min_raise,
            bet,
            player_bet,
            // bet_count,
            total_bet_count: total_raise_count,
            // raise_count,
            total_raise_count,
            // When creating from existing bets, we don't know which were forced
            forced_bet_count: 0,
            to_act_idx: to_act,
        }
    }

    pub fn advance_action(&mut self) {
        loop {
            // Here we use the length of the player bet vector
            // for the number of seats in the table. This assumes that
            // that the vector is always pre-initialized to the correct length.
            self.to_act_idx = (self.to_act_idx + 1) % self.player_bet.len();
            if self.needs_action.empty() || self.needs_action.get(self.to_act_idx) {
                break;
            }
        }
    }

    pub fn do_bet(&mut self, extra_amount: f32, is_forced: bool) {
        self.player_bet[self.to_act_idx] += extra_amount;
        self.total_bet_count += 1;

        if is_forced {
            self.forced_bet_count += 1;
        }

        // The amount to be called is
        // the maximum anyone has wagered.
        let previous_bet = self.bet;
        let player_bet = self.player_bet[self.to_act_idx];
        self.bet = previous_bet.max(player_bet);

        if !is_forced && player_bet > previous_bet {
            self.total_raise_count += 1;
        }

        let raise_amount = self.bet - previous_bet;
        self.min_raise = self.min_raise.max(raise_amount);
    }

    pub fn num_players_need_action(&self) -> usize {
        self.needs_action.count()
    }

    /// Returns true if no player has voluntarily put money in the pot this round.
    /// This means only forced bets (blinds, antes, straddles) have been posted.
    /// Returns true if no voluntary bets have been made yet (only forced bets like blinds/antes).
    pub fn is_action_unopened(&self) -> bool {
        self.total_bet_count == self.forced_bet_count
    }

    pub fn current_player_bet(&self) -> f32 {
        self.player_bet[self.to_act_idx]
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct GameState {
    /// The number of players that started
    pub num_players: usize,
    /// Which players are still active in the game.
    pub player_active: PlayerBitSet,
    pub player_all_in: PlayerBitSet,
    /// The total amount in all pots
    pub total_pot: f32,
    /// How much is left in each player's stack
    pub stacks: Vec<f32>,
    // The amount at the start of the game (or creation of the gamestate).
    pub starting_stacks: Vec<f32>,
    pub player_bet: Vec<f32>,
    pub player_winnings: Vec<f32>,
    /// The big blind size
    pub big_blind: f32,
    /// The small blind size
    pub small_blind: f32,
    /// The ante size
    pub ante: f32,
    /// The hands for each player. We keep hands
    /// even if the player is not currently active.
    pub hands: Vec<Hand>,
    /// The index of the player who's the dealer
    pub dealer_idx: usize,
    // What round this is currently
    pub round: Round,
    /// This is the round before we completed the game.
    /// Sometimes the game completes because of
    /// all the players fold in the preflop.
    pub round_before: Round,
    // ALl the current state of the round.
    pub round_data: RoundData,
    // The community cards.
    pub board: Vec<Card>,
    // Have the blinds been posted.
    // This is used to not double post blinds
    // on sim restarts.
    pub bb_posted: bool,
    pub sb_posted: bool,
    /// Maximum raises allowed per betting round. None = unlimited.
    /// Default is Some(3). When exceeded, raises are converted to calls.
    pub max_raises_per_round: Option<u8>,
}

impl GameState {
    /// Check if raises are capped for the current round.
    /// Returns true if the number of raises has reached the max limit.
    pub fn is_raise_capped(&self) -> bool {
        self.max_raises_per_round
            .is_some_and(|max| self.round_data.total_raise_count >= max)
    }

    pub fn num_active_players(&self) -> usize {
        self.player_active.count()
    }

    pub fn num_all_in_players(&self) -> usize {
        self.player_all_in.count()
    }

    pub fn is_complete(&self) -> bool {
        self.num_active_players() == 1 || self.round == Round::Complete
    }

    pub fn to_act_idx(&self) -> usize {
        self.round_data.to_act_idx
    }

    pub fn current_player_stack(&self) -> f32 {
        *self.stacks.get(self.to_act_idx()).unwrap_or(&0.0)
    }

    pub fn current_player_starting_stack(&self) -> f32 {
        *self.starting_stacks.get(self.to_act_idx()).unwrap_or(&0.0)
    }

    pub fn current_round_current_player_bet(&self) -> f32 {
        *self
            .round_data
            .player_bet
            .get(self.to_act_idx())
            .unwrap_or(&0.0)
    }

    pub fn current_round_bet(&self) -> f32 {
        self.round_data.bet
    }

    pub fn current_round_player_bet(&self, idx: usize) -> f32 {
        self.round_data.player_bet.get(idx).copied().unwrap_or(0.0)
    }

    pub fn current_round_num_active_players(&self) -> usize {
        self.round_data.num_players_need_action()
    }

    pub fn current_round_min_raise(&self) -> f32 {
        self.round_data.min_raise
    }

    pub fn advance_round(&mut self) {
        match self.round {
            Round::Complete => (),
            _ => self.advance_normal(),
        }
    }

    fn advance_normal(&mut self) {
        // We're advancing (not completing) so
        // keep advanding the round_before field as well.
        self.round_before = self.round;

        self.round = self.round.advance();

        let mut round_data = RoundData::new(
            self.num_players,
            self.big_blind,
            self.player_active,
            self.dealer_idx,
        );
        round_data.advance_action();
        if self.round == Round::Preflop && self.num_players == 2 {
            // With only two players, it is the dealer that has
            // to post the small blind, so pass the action back.
            round_data.advance_action();
        }
        self.round_data = round_data;
    }

    pub fn complete(&mut self) {
        if self.round == Round::Complete {
            return;
        }

        self.round_before = self.round;
        self.round = Round::Complete;
        self.round_data = RoundData::new(
            self.num_players,
            self.big_blind,
            PlayerBitSet::new(0),
            self.dealer_idx,
        );
    }

    pub fn fold(&mut self) {
        // Which player is next to act
        let idx = self.round_data.to_act_idx;
        // We are going to change the current round since this player is out.
        self.round_data.needs_action.disable(idx);
        self.player_active.disable(idx);

        // They fold ending the turn.
        self.round_data.advance_action();
    }

    pub fn do_bet(&mut self, amount: f32, is_forced: bool) -> Result<f32, GameStateError> {
        // Which player is next to act
        let idx = self.to_act_idx();

        // This is the amount extra that the player is putting into the round's betting
        // pot
        //
        // We need to validate it before making anychanges to the game state. This
        // allows us to return an error before getting into any bad gamestate.
        //
        // It also allows agents to be punished for putting in bad bet types.
        //
        // Make sure the bet is a correct amount and if not
        // then cap it at the maximum the player can bet (Their stacks usually)
        let extra_amount = if is_forced {
            self.validate_forced_bet_amount(amount)
        } else {
            self.validate_bet_amount(amount)?
        };

        let prev_bet = self.round_data.bet;
        // At this point we start making changes.
        // Take the money out.
        self.stacks[idx] -= extra_amount;

        self.round_data.do_bet(extra_amount, is_forced);

        self.player_bet[idx] += extra_amount;

        self.total_pot += extra_amount;

        let is_betting_reopened = prev_bet < self.round_data.bet;

        if is_betting_reopened {
            // This is a new max bet. We need to reset who can act in the round
            self.round_data.needs_action = self.player_active;
        }

        // If they put money into the pot then they are done this turn.
        if !is_forced {
            self.round_data.needs_action.disable(idx);
        }

        // We're out and can't continue
        // Use epsilon comparison to handle floating-point precision issues
        // (e.g., when stack is 1.19e-7 instead of exactly 0)
        if abs_diff_eq!(self.stacks[idx], 0.0) {
            // Keep track of who's still active.
            self.player_active.disable(idx);
            // Keep track of going all in. We'll use that later on
            // to determine who's worth ranking.
            self.player_all_in.enable(idx);
            // It doesn' matter if this is a forced
            // bet if the player is out of money.
            self.round_data.needs_action.disable(idx);
        }

        // Advance the next to act.
        self.round_data.advance_action();

        Ok(extra_amount)
    }

    pub fn award(&mut self, player_idx: usize, amount: f32) {
        self.stacks[player_idx] += amount;
        self.player_winnings[player_idx] += amount;
    }

    /// Get the total reward for a player.
    /// This is the change in stack from the start of the game
    /// to the now.
    ///
    /// # Arguments
    /// * `player_idx` - The index of the player to get the reward for.
    pub fn player_reward(&self, player_idx: usize) -> f32 {
        // The reward is the change in stack from the start of the game
        // to the end of the game.
        self.stacks[player_idx] - self.starting_stacks[player_idx]
    }

    fn validate_forced_bet_amount(&self, amount: f32) -> f32 {
        // Which player is next to act. Map the optional into the to_act_index or 0.
        let idx = self.to_act_idx();

        self.stacks[idx].min(amount)
    }

    fn validate_bet_amount(&self, amount: f32) -> Result<f32, GameStateError> {
        // Which player is next to act
        let idx = self.to_act_idx();

        // Use a scaled epsilon for floating point comparisons.
        // We scale by the magnitude of the values being compared to handle
        // both small and large bet amounts correctly.
        let magnitude = amount.abs().max(self.round_data.bet.abs()).max(1.0);
        let epsilon = magnitude * f32::EPSILON * 1000.0; // Scale epsilon for practical tolerance

        if amount.is_sign_negative() || amount.is_nan() {
            // You can't bet negative numbers.
            // You can't be a NaN.
            Err(GameStateError::BetInvalidSize)
        } else if self.round_data.player_bet[idx] > amount + epsilon {
            // We've already bet more than this. No takes backs.
            Err(GameStateError::BetSizeDoesntCallSelf)
        } else {
            // How much extra are we putting in.
            let extra = amount - self.round_data.player_bet[idx];

            // How much more are we putting in this time. Capped at the stack
            let capped_extra = self.stacks[idx].min(extra);
            // What our new player bet will be
            let capped_new_player_bet = self.round_data.player_bet[idx] + capped_extra;
            let current_bet = self.round_data.bet;
            // How much this is a raise.
            let raise = (capped_new_player_bet - current_bet).max(0.0);
            // Use relative epsilon for all-in check to handle floating point precision
            let stack_epsilon = self.stacks[idx].abs().max(1.0) * f32::EPSILON * 1000.0;
            let is_all_in = (capped_extra - self.stacks[idx]).abs() < stack_epsilon;
            let is_raise = raise > epsilon;
            // Use epsilon tolerance for call check to handle floating point precision
            if capped_new_player_bet + epsilon < self.round_data.bet && !is_all_in {
                // If we're not even calling and it's not an all in.
                Err(GameStateError::BetSizeDoesntCall)
            } else if is_raise && !is_all_in && raise + epsilon < self.round_data.min_raise {
                // There's a raise the raise is less than the min bet and it's not an all in
                Err(GameStateError::RaiseSizeTooSmall)
            } else {
                // Yeah this looks ok.
                Ok(capped_extra)
            }
        }
    }
}

pub trait GameStateGenerator: Iterator<Item = GameState> {}

/// This is a simple generator that just clones the game state
/// every time it's called.
///
/// This holds the dealt cards constant and the stack sizes constant.
pub struct CloneGameStateGenerator {
    game_state: GameState,
}

impl CloneGameStateGenerator {
    pub fn new(game_state: GameState) -> CloneGameStateGenerator {
        CloneGameStateGenerator { game_state }
    }
}

impl Iterator for CloneGameStateGenerator {
    type Item = GameState;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.game_state.clone())
    }
}

/// This `GameStateGenerator` generates a random game state with no cards dealt
/// and random stack sizes. The dealer button is also randomly placed.
pub struct RandomGameStateGenerator {
    num_players: usize,
    min_stack: f32,
    max_stack: f32,
    big_blind: f32,
    small_blind: f32,
    ante: f32,
    /// Optional seeded RNG for deterministic generation
    seeded_rng: Option<rand::rngs::StdRng>,
}

impl RandomGameStateGenerator {
    pub fn new(
        num_players: usize,
        min_stack: f32,
        max_stack: f32,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
    ) -> RandomGameStateGenerator {
        RandomGameStateGenerator {
            num_players,
            min_stack,
            max_stack,
            big_blind,
            small_blind,
            ante,
            seeded_rng: None,
        }
    }

    /// Create a new generator with a specific seed for deterministic results
    pub fn with_seed(
        num_players: usize,
        min_stack: f32,
        max_stack: f32,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
        seed: u64,
    ) -> RandomGameStateGenerator {
        use rand::SeedableRng;
        RandomGameStateGenerator {
            num_players,
            min_stack,
            max_stack,
            big_blind,
            small_blind,
            ante,
            seeded_rng: Some(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }
}

impl Iterator for RandomGameStateGenerator {
    type Item = GameState;

    fn next(&mut self) -> Option<Self::Item> {
        use rand::Rng;

        // Use seeded RNG if available, otherwise use global RNG
        let (stacks, dealer_idx) = if let Some(ref mut seeded) = self.seeded_rng {
            let stacks: Vec<f32> = (0..self.num_players)
                .map(|_| {
                    if self.min_stack == self.max_stack {
                        self.min_stack
                    } else {
                        seeded.random_range(self.min_stack..self.max_stack)
                    }
                })
                .collect();
            let dealer_idx = seeded.random_range(0..self.num_players);
            (stacks, dealer_idx)
        } else {
            let mut unseeded = rng();
            let stacks: Vec<f32> = (0..self.num_players)
                .map(|_| {
                    if self.min_stack == self.max_stack {
                        self.min_stack
                    } else {
                        unseeded.random_range(self.min_stack..self.max_stack)
                    }
                })
                .collect();
            let dealer_idx = unseeded.random_range(0..self.num_players);
            (stacks, dealer_idx)
        };

        Some(
            GameStateBuilder::new()
                .stacks(stacks)
                .big_blind(self.big_blind)
                .small_blind(self.small_blind)
                .ante(self.ante)
                .dealer_idx(dealer_idx)
                .build()
                .expect("RandomGameStateGenerator produced invalid game state"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper to create a game state with standard defaults
    fn test_game_state(
        stacks: Vec<f32>,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
        dealer_idx: usize,
    ) -> GameState {
        GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(big_blind)
            .small_blind(small_blind)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap()
    }

    #[test]
    fn test_fold_around_call() {
        let stacks = vec![100.0; 4];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 1);

        // starting
        game_state.advance_round();
        // Ante
        game_state.advance_round();
        // Deal Preflop
        game_state.advance_round();

        // Preflop
        // 0 player, 1 dealer, 2 small blind, 3 big blind
        // Game state doesn't force the small blind and big blind
        assert_eq!(2, game_state.to_act_idx());

        // Do the blinds now
        game_state.do_bet(5.0, true).unwrap();
        game_state.do_bet(10.0, true).unwrap();

        // The blinds posting wraps around when needed
        assert_eq!(0, game_state.to_act_idx());

        // Posted blinds can then fold
        game_state.fold();
        game_state.fold();

        game_state.do_bet(10.0, false).unwrap();
        game_state.do_bet(10.0, false).unwrap();
        assert_eq!(0, game_state.current_round_num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // Deal  Flop
        game_state.advance_round();

        // Flop
        game_state.advance_round();
        assert_eq!(2, game_state.to_act_idx());
        game_state.do_bet(0.0, false).unwrap();
        assert_eq!(3, game_state.to_act_idx());
        game_state.do_bet(0.0, false).unwrap();
        assert_eq!(0, game_state.current_round_num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // Deal Turn
        game_state.advance_round();

        // Turn
        game_state.advance_round();
        assert_eq!(2, game_state.to_act_idx());
        assert_eq!(2, game_state.current_round_num_active_players());
        game_state.do_bet(0.0, false).unwrap();
        game_state.do_bet(0.0, false).unwrap();
        assert_eq!(0, game_state.current_round_num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // Deal River
        game_state.advance_round();

        // River
        game_state.advance_round();
        game_state.do_bet(0.0, false).unwrap();
        game_state.do_bet(0.0, false).unwrap();
        assert_eq!(0, game_state.current_round_num_active_players());
        assert_eq!(2, game_state.num_active_players());

        game_state.advance_round();
        assert_eq!(Round::Showdown, game_state.round);
    }

    #[test]
    fn test_cant_bet_less_0() {
        let stacks = vec![100.0; 5];
        let mut game_state = test_game_state(stacks, 2.0, 1.0, 0.0, 0);
        game_state.advance_round();
        game_state.advance_round();

        game_state.do_bet(33.0, false).unwrap();
        game_state.fold();
        let res = game_state.do_bet(20.0, false);

        assert_eq!(res.err(), Some(GameStateError::BetSizeDoesntCall));
    }

    #[test]
    fn test_cant_bet_less_with_all_in() {
        let stacks = vec![100.0, 50.0, 50.0, 100.0, 10.0];
        let mut game_state = test_game_state(stacks, 2.0, 1.0, 0.0, 0);
        // Do the start and ante rounds and setup next to act
        game_state.advance_round();
        game_state.advance_round();

        // UTG raises to 10
        game_state.do_bet(10.0, false).unwrap();

        // UTG+1 has 10 remaining so betting 100 is overbetting
        // into an all in.
        game_state.do_bet(100.0, false).unwrap();

        // Dealer gets out of the way
        game_state.fold();

        // Small Blind raises to 20
        game_state.do_bet(20.0, false).unwrap();

        // Big Blind can't call the previous value.
        let res = game_state.do_bet(10.0, false);
        assert_eq!(res.err(), Some(GameStateError::BetSizeDoesntCall));
    }

    #[test]
    fn test_cant_under_minraise_bb() {
        let stacks = vec![500.0; 5];
        let mut game_state = test_game_state(stacks, 20.0, 10.0, 0.0, 0);
        // Do the start and ante rounds and setup next to act
        game_state.advance_round();
        game_state.advance_round();
        game_state.advance_round();

        game_state.do_bet(10.0, true).unwrap();
        game_state.do_bet(20.0, true).unwrap();

        // UTG raises to 33
        //
        // However the min raise is the big blind
        // so since the bb has already posted
        // we're not able to raise 13
        assert_eq!(
            Err(GameStateError::RaiseSizeTooSmall),
            game_state.do_bet(33.0, false)
        );
    }

    #[test]
    fn test_gamestate_keeps_round_before_complete() {
        let stacks = vec![100.0; 3];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        // Simulate a game where everyone folds and the big blind wins
        game_state.advance_round();
        game_state.advance_round();
        game_state.advance_round();
        game_state.fold();
        game_state.fold();
        game_state.complete();
        assert_eq!(Round::Complete, game_state.round);
        assert_eq!(Round::Preflop, game_state.round_before);
    }

    #[test]
    fn test_can_create_starting_round_data() {
        let num_players = 3;
        let min_raise = 10.0;
        let active = PlayerBitSet::new(num_players);

        let round_data = RoundData::new(num_players, min_raise, active, 0);

        assert_eq!(round_data.bet, 0.0);

        assert_eq!(round_data.total_bet_count, 0);

        assert_eq!(round_data.total_raise_count, 0);
    }

    #[test]
    fn test_can_create_inprogress_round_data() {
        let num_players = 3;
        let min_raise = 10.0;
        let active = PlayerBitSet::new(num_players);

        let player_bet = vec![0.0, 10.0, 20.0];
        let to_act = 0;

        let round_data = RoundData::new_with_bets(min_raise, active, to_act, player_bet);

        assert_eq!(round_data.bet, 20.0);

        assert_eq!(round_data.total_bet_count, 2);

        assert_eq!(round_data.total_raise_count, 2);
    }

    /// Verifies that each Round variant displays the expected string representation.
    #[test]
    fn test_round_display() {
        assert_eq!(format!("{}", Round::Starting), "Starting");
        assert_eq!(format!("{}", Round::Ante), "Ante");
        assert_eq!(format!("{}", Round::DealPreflop), "Deal Preflop");
        assert_eq!(format!("{}", Round::Preflop), "Preflop");
        assert_eq!(format!("{}", Round::DealFlop), "Deal Flop");
        assert_eq!(format!("{}", Round::Flop), "Flop");
        assert_eq!(format!("{}", Round::DealTurn), "Deal Turn");
        assert_eq!(format!("{}", Round::Turn), "Turn");
        assert_eq!(format!("{}", Round::DealRiver), "Deal River");
        assert_eq!(format!("{}", Round::River), "River");
        assert_eq!(format!("{}", Round::Showdown), "Showdown");
        assert_eq!(format!("{}", Round::Complete), "Complete");
    }

    /// Verifies is_action_unopened correctly tracks whether voluntary bets have been made.
    #[test]
    fn test_is_action_unopened() {
        let num_players = 3;
        let active = PlayerBitSet::new(num_players);
        let mut round_data = RoundData::new(num_players, 10.0, active, 0);

        // Initially, no bets have been made
        assert!(round_data.is_action_unopened());

        // After a forced bet (like a blind), still unopened
        round_data.do_bet(5.0, true);
        assert!(round_data.is_action_unopened());

        // After a voluntary bet, no longer unopened
        round_data.advance_action();
        round_data.do_bet(10.0, false);
        assert!(!round_data.is_action_unopened());
    }

    /// Verifies current_player_bet returns the actual bet amount for the current player.
    #[test]
    fn test_current_player_bet() {
        let num_players = 3;
        let active = PlayerBitSet::new(num_players);
        let mut round_data = RoundData::new(num_players, 10.0, active, 0);

        // Initially 0
        assert_eq!(round_data.current_player_bet(), 0.0);

        // After betting
        round_data.do_bet(25.0, false);
        assert_eq!(round_data.current_player_bet(), 25.0);

        // Move to next player and check their bet
        round_data.advance_action();
        assert_eq!(round_data.current_player_bet(), 0.0);

        round_data.do_bet(50.0, false);
        assert_eq!(round_data.current_player_bet(), 50.0);
    }

    /// Verifies num_all_in_players correctly counts players who have gone all-in.
    #[test]
    fn test_num_all_in_players() {
        let stacks = vec![100.0, 100.0, 100.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);

        // Initially no one is all-in
        assert_eq!(game_state.num_all_in_players(), 0);

        // Advance to preflop
        game_state.advance_round();
        game_state.advance_round();
        game_state.advance_round();
        assert_eq!(game_state.num_all_in_players(), 0);

        // Go all-in
        game_state.do_bet(100.0, false).unwrap();
        assert_eq!(game_state.num_all_in_players(), 1);

        // Another player goes all-in
        game_state.do_bet(100.0, false).unwrap();
        assert_eq!(game_state.num_all_in_players(), 2);
    }

    /// Verifies is_complete correctly identifies when a game has ended.
    #[test]
    fn test_is_complete() {
        let stacks = vec![100.0, 100.0, 100.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);

        // Not complete at start
        assert!(!game_state.is_complete());

        // Advance through rounds
        game_state.advance_round();
        game_state.advance_round();
        game_state.advance_round();
        assert!(!game_state.is_complete());

        // After two folds, only one player remains - should be complete
        game_state.fold();
        game_state.fold();
        assert!(game_state.is_complete());
    }

    /// Verifies do_bet correctly updates bet amounts, counts, and min_raise tracking.
    #[test]
    fn test_do_bet_arithmetic() {
        let num_players = 3;
        let active = PlayerBitSet::new(num_players);
        let mut round_data = RoundData::new(num_players, 10.0, active, 0);

        // First bet of 20
        round_data.do_bet(20.0, false);
        assert_eq!(round_data.player_bet[0], 20.0);
        assert_eq!(round_data.bet, 20.0);
        assert_eq!(round_data.total_bet_count, 1);
        assert_eq!(round_data.total_raise_count, 1);

        // Advance and second player raises to 40
        round_data.advance_action();
        round_data.do_bet(40.0, false);
        assert_eq!(round_data.player_bet[1], 40.0);
        assert_eq!(round_data.bet, 40.0);
        assert_eq!(round_data.total_bet_count, 2);
        assert_eq!(round_data.total_raise_count, 2);
        assert_eq!(round_data.min_raise, 20.0); // 40 - 20 = 20

        // Test forced bet tracking
        let mut round_data2 = RoundData::new(num_players, 10.0, active, 0);
        round_data2.do_bet(5.0, true); // forced
        assert_eq!(round_data2.forced_bet_count, 1);
        assert_eq!(round_data2.total_raise_count, 0); // forced bets don't count as raises
    }

    /// Verifies current_player_starting_stack returns the player's initial stack amount.
    #[test]
    fn test_current_player_starting_stack() {
        let stacks = vec![100.0, 200.0, 300.0];
        let game_state = test_game_state(stacks.clone(), 10.0, 5.0, 0.0, 0);

        // Player 1 is to act first (after dealer at position 0)
        let starting_stack = game_state.current_player_starting_stack();
        // Starting stacks should be what we passed in
        assert_eq!(starting_stack, stacks[game_state.to_act_idx()]);
        assert!(starting_stack > 1.0, "Starting stack should be > 1.0");
        assert!(starting_stack > 0.0, "Starting stack should be > 0.0");
    }

    /// Verifies current_round_current_player_bet returns what the current player has bet this round.
    #[test]
    fn test_current_round_current_player_bet() {
        let stacks = vec![100.0, 200.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round(); // Move to preflop

        // Post blinds
        let _ = game_state.do_bet(5.0, true); // Small blind
        let _player_bet = game_state.current_round_current_player_bet();

        // After posting SB, player should have bet 5.0
        // Now it's BB's turn
        let bb_bet = game_state.current_round_current_player_bet();
        // BB hasn't bet yet
        assert_eq!(bb_bet, 0.0);

        // Post BB
        let _ = game_state.do_bet(10.0, true);
        // Now SB's turn again
        let sb_current_bet = game_state.current_round_current_player_bet();
        // SB has bet 5.0
        assert_eq!(sb_current_bet, 5.0);
    }

    /// Verifies advance_round is a no-op when the game is already complete.
    #[test]
    fn test_advance_round_when_complete() {
        let stacks = vec![100.0, 100.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        game_state.complete();

        assert_eq!(game_state.round, Round::Complete);
        let round_before = game_state.round_before;

        // Calling advance_round when complete should be a no-op
        game_state.advance_round();

        assert_eq!(game_state.round, Round::Complete);
        assert_eq!(game_state.round_before, round_before);
    }

    /// Verifies validate_bet_amount rejects negative and NaN amounts.
    #[test]
    fn test_validate_bet_amount_negative() {
        let stacks = vec![100.0, 100.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        let result = game_state.validate_bet_amount(-10.0);
        assert!(result.is_err());

        let nan_result = game_state.validate_bet_amount(f32::NAN);
        assert!(nan_result.is_err());
    }

    /// Verifies RandomGameStateGenerator produces game states with valid properties.
    #[test]
    fn test_random_game_state_generator() {
        let mut generator = RandomGameStateGenerator::new(3, 50.0, 150.0, 10.0, 5.0, 0.0);

        for _ in 0..10 {
            let gs = generator.next().unwrap();
            assert_eq!(gs.num_players, 3);
            assert!(gs.dealer_idx < gs.num_players);
            for stack in &gs.stacks {
                assert!(*stack >= 50.0 && *stack <= 150.0);
            }
        }
    }

    // ==================== GameStateBuilder Tests ====================

    #[test]
    fn test_builder_minimal_valid() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.num_players, 2);
        assert_eq!(gs.big_blind, 10.0);
        assert_eq!(gs.small_blind, 5.0); // defaults to bb/2
        assert_eq!(gs.ante, 0.0);
        assert_eq!(gs.dealer_idx, 0);
        assert_eq!(gs.round, Round::Starting);
        assert_eq!(gs.max_raises_per_round, Some(3));
    }

    #[test]
    fn test_builder_small_blind_computed_from_big_blind() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(20.0)
            .build()
            .unwrap();

        assert_eq!(gs.small_blind, 10.0);
    }

    #[test]
    fn test_builder_num_players_with_stack_convenience() {
        let gs = GameStateBuilder::new()
            .num_players_with_stack(4, 500.0)
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.num_players, 4);
        assert_eq!(gs.stacks, vec![500.0; 4]);
    }

    #[test]
    fn test_builder_max_raises_default_is_three() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.max_raises_per_round, Some(3));
    }

    #[test]
    fn test_builder_max_raises_unlimited() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .max_raises_per_round(None)
            .build()
            .unwrap();

        assert_eq!(gs.max_raises_per_round, None);
    }

    #[test]
    fn test_builder_blinds_convenience_method() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .blinds(20.0, 10.0)
            .build()
            .unwrap();

        assert_eq!(gs.big_blind, 20.0);
        assert_eq!(gs.small_blind, 10.0);
    }

    #[test]
    fn test_builder_error_missing_stacks() {
        let result = GameStateBuilder::new().big_blind(10.0).build();

        assert_eq!(result.unwrap_err(), GameStateBuilderError::MissingStacks);
    }

    #[test]
    fn test_builder_error_missing_big_blind() {
        let result = GameStateBuilder::new().stacks(vec![100.0, 100.0]).build();

        assert_eq!(result.unwrap_err(), GameStateBuilderError::MissingBigBlind);
    }

    #[test]
    fn test_builder_error_too_few_players() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0])
            .big_blind(10.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidPlayerCount { actual: 1 }
        );
    }

    #[test]
    fn test_builder_error_too_many_players() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0; 17])
            .big_blind(10.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidPlayerCount { actual: 17 }
        );
    }

    #[test]
    fn test_builder_error_invalid_dealer_index() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .dealer_idx(5)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidDealerIndex {
                dealer_idx: 5,
                num_players: 2
            }
        );
    }

    #[test]
    fn test_builder_error_negative_big_blind() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(-10.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidBigBlind(-10.0)
        );
    }

    #[test]
    fn test_builder_error_zero_big_blind() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(0.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidBigBlind(0.0)
        );
    }

    #[test]
    fn test_builder_error_nan_big_blind() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(f32::NAN)
            .build();

        // NaN != NaN, so we check the variant
        assert!(matches!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidBigBlind(_)
        ));
    }

    #[test]
    fn test_builder_error_negative_small_blind() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .small_blind(-5.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidSmallBlind(-5.0)
        );
    }

    #[test]
    fn test_builder_error_negative_ante() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .ante(-1.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidAnte(-1.0)
        );
    }

    #[test]
    fn test_builder_error_negative_stack() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, -50.0])
            .big_blind(10.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidStack {
                index: 1,
                value: -50.0
            }
        );
    }

    #[test]
    fn test_builder_error_insufficient_active_players() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 0.0])
            .big_blind(10.0)
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InsufficientActivePlayers
        );
    }

    #[test]
    fn test_builder_error_hands_length_mismatch() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .hands(vec![Hand::default()])
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::HandsLengthMismatch {
                hands_len: 1,
                num_players: 2
            }
        );
    }

    #[test]
    fn test_builder_error_player_bet_length_mismatch() {
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .player_bet(vec![0.0, 0.0, 0.0])
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::PlayerBetLengthMismatch {
                bet_len: 3,
                num_players: 2
            }
        );
    }

    #[test]
    fn test_builder_error_invalid_board_size_one() {
        use crate::core::{Card, Suit, Value};
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .board(vec![Card::new(Value::Ace, Suit::Spade)])
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidBoardSize(1)
        );
    }

    #[test]
    fn test_builder_error_invalid_board_size_two() {
        use crate::core::{Card, Suit, Value};
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .board(vec![
                Card::new(Value::Ace, Suit::Spade),
                Card::new(Value::King, Suit::Spade),
            ])
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::InvalidBoardSize(2)
        );
    }

    #[test]
    fn test_builder_error_duplicate_card_in_board() {
        use crate::core::{Card, Suit, Value};
        let card = Card::new(Value::Ace, Suit::Spade);
        let result = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .board(vec![card, Card::new(Value::King, Suit::Spade), card])
            .build();

        assert_eq!(
            result.unwrap_err(),
            GameStateBuilderError::DuplicateCard(card)
        );
    }

    #[test]
    fn test_builder_heads_up_valid() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.num_players, 2);
    }

    #[test]
    fn test_builder_ten_players_valid() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0; 10])
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.num_players, 10);
    }

    #[test]
    fn test_builder_all_but_two_players_zero_stack() {
        let mut stacks = vec![0.0; 6];
        stacks[2] = 100.0;
        stacks[5] = 100.0;

        let gs = GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(10.0)
            .build()
            .unwrap();

        assert_eq!(gs.num_players, 6);
        assert_eq!(gs.player_active.count(), 2);
    }

    #[test]
    fn test_builder_with_ante() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .ante(1.0)
            .build()
            .unwrap();

        assert_eq!(gs.ante, 1.0);
    }

    #[test]
    fn test_builder_with_dealer_idx() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0, 100.0])
            .big_blind(10.0)
            .dealer_idx(2)
            .build()
            .unwrap();

        assert_eq!(gs.dealer_idx, 2);
    }

    #[test]
    fn test_builder_with_round() {
        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .round(Round::Preflop)
            .build()
            .unwrap();

        assert_eq!(gs.round, Round::Preflop);
        // When not Starting, blinds are marked as posted
        assert!(gs.bb_posted);
        assert!(gs.sb_posted);
    }

    #[test]
    fn test_builder_with_valid_board() {
        use crate::core::{Card, Suit, Value};
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
        ];

        let gs = GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .big_blind(10.0)
            .board(board.clone())
            .build()
            .unwrap();

        assert_eq!(gs.board, board);
    }
}
