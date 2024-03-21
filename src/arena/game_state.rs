use core::fmt;
use std::fmt::Display;

use crate::core::{Card, Hand, PlayerBitSet, Rank};

use super::errors::GameStateError;

/// The round of the game.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
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

#[derive(Clone, PartialEq)]
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
    // The number of times the player has put in money.
    pub bet_count: Vec<u8>,
    // The number of times anyone has put in money
    pub total_bet_count: u8,
    // The number of times the player has increased the bet voluntarily.
    pub raise_count: Vec<u8>,
    // The number of times anyone has increased the bet non-forced.
    pub total_raise_count: u8,
    // The index of the next player to act.
    pub to_act_idx: usize,
    // The computed rank of the player's hand at showdown.
    // This will be `None` if there's no showdown, or
    // for other rounds
    pub hand_rank: Vec<Option<Rank>>,
}

impl RoundData {
    pub fn new(num_players: usize, min_raise: f32, active: PlayerBitSet, to_act: usize) -> Self {
        RoundData {
            needs_action: active,
            starting_player_active: active,
            min_raise,
            bet: 0.0,
            player_bet: vec![0.0; num_players],
            bet_count: vec![0; num_players],
            total_bet_count: 0,
            raise_count: vec![0; num_players],
            total_raise_count: 0,
            to_act_idx: to_act,
            hand_rank: vec![None; num_players],
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
        self.bet_count[self.to_act_idx] += 1;
        self.total_bet_count += 1;

        // The amount to be called is
        // the maximum anyone has wagered.
        let previous_bet = self.bet;
        let player_bet = self.player_bet[self.to_act_idx];
        self.bet = previous_bet.max(player_bet);

        if !is_forced && player_bet > previous_bet {
            self.raise_count[self.to_act_idx] += 1;
            self.total_raise_count += 1;
        }

        let raise_amount = self.bet - previous_bet;
        self.min_raise = self.min_raise.max(raise_amount);
    }

    pub fn num_players_need_action(&self) -> usize {
        self.needs_action.count()
    }

    pub fn current_player_bet(&self) -> f32 {
        self.player_bet[self.to_act_idx]
    }

    pub fn set_hand_rank(&mut self, player_idx: usize, rank: Rank) {
        match &self.hand_rank[player_idx] {
            Some(r) => self.hand_rank[player_idx] = Some(rank.max(*r)),
            None => self.hand_rank[player_idx] = Some(rank),
        }
    }
}

#[derive(Clone, PartialEq)]
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
    /// If there was a showdown then we'll have the
    /// computed rank of the player's hand.
    pub computed_rank: Vec<Option<Rank>>,
    /// The index of the player who's the dealer
    pub dealer_idx: usize,
    // What round this is currently
    pub round: Round,
    // ALl the current state of the round.
    pub round_data: RoundData,
    // The community cards.
    pub board: Vec<Card>,
    // Have the blinds been posted.
    // This is used to not double post blinds
    // on sim restarts.
    pub bb_posted: bool,
    pub sb_posted: bool,
}

impl GameState {
    pub fn new(
        stacks: Vec<f32>,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
        dealer_idx: usize,
    ) -> Self {
        let num_players = stacks.len();
        GameState {
            num_players,
            starting_stacks: stacks.clone(),
            stacks,
            big_blind,
            small_blind,
            ante,
            player_active: PlayerBitSet::new(num_players),
            player_all_in: PlayerBitSet::default(),
            player_bet: vec![0.0; num_players],
            player_winnings: vec![0.0; num_players],
            dealer_idx,
            total_pot: 0.0,
            hands: vec![Hand::default(); num_players],
            round: Round::Starting,
            board: vec![],
            round_data: RoundData::new(
                num_players,
                big_blind,
                PlayerBitSet::new(num_players),
                dealer_idx,
            ),
            computed_rank: vec![None; num_players],
            bb_posted: false,
            sb_posted: false,
        }
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
        self.round = Round::Complete;
        let round_data = RoundData::new(
            self.num_players,
            self.big_blind,
            PlayerBitSet::new(0),
            self.dealer_idx,
        );
        self.round_data = round_data;
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
        if self.stacks[idx] <= 0.0 {
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

    fn validate_forced_bet_amount(&self, amount: f32) -> f32 {
        // Which player is next to act. Map the optional into the to_act_index or 0.
        let idx = self.to_act_idx();

        self.stacks[idx].min(amount)
    }

    fn validate_bet_amount(&self, amount: f32) -> Result<f32, GameStateError> {
        // Which player is next to act
        let idx = self.to_act_idx();

        if amount.is_sign_negative() || amount.is_nan() {
            // You can't bet negative numbers.
            // You can't be a NaN.
            Err(GameStateError::BetInvalidSize)
        } else if self.round_data.player_bet[idx] > amount {
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
            let is_all_in = capped_extra == self.stacks[idx];
            let is_raise = raise > 0.0;
            if capped_new_player_bet < self.round_data.bet && !is_all_in {
                // If we're not even calling and it's not an all in.
                Err(GameStateError::BetSizeDoesntCall)
            } else if is_raise && !is_all_in && raise < self.round_data.min_raise {
                // There's a raise the raise is less than the min bet and it's not an all in
                Err(GameStateError::RaiseSizeTooSmall)
            } else {
                // Yeah this looks ok.
                Ok(capped_extra)
            }
        }
    }
}

impl fmt::Debug for RoundData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoundData")
            .field("needs_action", &self.needs_action)
            .field("num_players_need_action", &self.num_players_need_action())
            .field("min_raise", &self.min_raise)
            .field("bet", &self.bet)
            .field("player_bet", &self.player_bet)
            .field("bet_count", &self.bet_count)
            .field("raise_count", &self.raise_count)
            .field("to_act_idx", &self.to_act_idx)
            .finish()
    }
}

impl fmt::Debug for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GameState")
            .field("num_players", &self.num_players)
            .field("num_active_players", &self.num_active_players())
            .field("player_active", &self.player_active)
            .field("player_all_in", &self.player_all_in)
            .field("total_pot", &self.total_pot)
            .field("stacks", &self.stacks)
            .field("big_blind", &self.big_blind)
            .field("small_blind", &self.small_blind)
            .field("ante", &self.ante)
            .field("hands", &self.hands)
            .field("dealer_idx", &self.dealer_idx)
            .field("round", &self.round)
            .field("round_data", &self.round_data)
            .field("board", &self.board)
            .field("sb_posted", &self.sb_posted)
            .field("bb_posted", &self.bb_posted)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_around_call() {
        let stacks = vec![100.0; 4];
        let mut game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 1);

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
        let mut game_state = GameState::new(stacks, 2.0, 1.0, 0.0, 0);
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
        let mut game_state = GameState::new(stacks, 2.0, 1.0, 0.0, 0);
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
        let mut game_state = GameState::new(stacks, 20.0, 10.0, 0.0, 0);
        // Do the start and ante rounds and setup next to act
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
}
