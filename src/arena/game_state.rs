use core::fmt;

use crate::core::{Card, Hand, PlayerBitSet};

use super::errors::GameStateError;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Round {
    Starting,
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
    Complete,
}

impl Round {
    pub fn advance(&self) -> Self {
        match *self {
            Round::Starting => Round::Preflop,
            Round::Preflop => Round::Flop,
            Round::Flop => Round::Turn,
            Round::Turn => Round::River,
            Round::River => Round::Showdown,
            Round::Showdown => Round::Complete,
            Round::Complete => Round::Complete,
        }
    }
}

#[derive(Clone)]
pub struct RoundData {
    pub player_active: PlayerBitSet,
    // The minimum allowed bet.
    pub min_raise: i32,
    // The value to be called.
    pub bet: i32,
    // How much each player has put in so far.
    pub player_bet: Vec<i32>,
    // The number of times the player has put in money.
    pub bet_count: Vec<u8>,
    // The number of times the player has increased the bet voluntarily.
    pub raise_count: Vec<u8>,
    // The index of the next player to act.
    pub to_act_idx: usize,
}

impl RoundData {
    pub fn advance(&mut self) {
        loop {
            self.to_act_idx = (self.to_act_idx + 1) % self.player_bet.len();
            if self.player_active.empty() || self.player_active.get(self.to_act_idx) {
                break;
            }
        }
    }

    pub fn do_bet(&mut self, extra_ammount: i32, is_forced: bool) {
        self.player_bet[self.to_act_idx] += extra_ammount;
        self.bet_count[self.to_act_idx] += 1;

        // Before resetting check if this is a raise to keep count
        let is_bet = self.player_bet[self.to_act_idx] > self.bet;
        if !is_forced && is_bet {
            self.raise_count[self.to_act_idx] += 1;
        }

        // The amount to be called is
        // the maximum anyone has wagered.
        self.bet = self.bet.max(self.player_bet[self.to_act_idx]);

        // Keep the maximum bet ammount. Anything
        // smaller should be due to going all in.
        self.min_raise = self.min_raise.max(extra_ammount);
    }

    pub fn num_active_players(&self) -> usize {
        self.player_active.count()
    }

    pub fn current_player_bet(&self) -> i32 {
        self.player_bet[self.to_act_idx]
    }
}

#[derive(Clone)]
pub struct GameState {
    /// The number of players that started
    pub num_players: usize,
    /// Which players are still active in the game.
    pub player_active: PlayerBitSet,
    pub player_all_in: PlayerBitSet,
    /// The total ammount in all pots
    pub total_pot: i32,
    /// How much is left in each player's stack
    pub stacks: Vec<i32>,
    pub player_bet: Vec<i32>,
    pub player_winnings: Vec<i32>,
    /// The big blind size
    pub big_blind: i32,
    /// The small blind size
    pub small_blind: i32,
    /// The hands for each player. We keep hands
    /// even if the player is not currently active.
    pub hands: Vec<Hand>,
    /// The index of the player who's the dealer
    pub dealer_idx: usize,
    // What round this is currently
    pub round: Round,
    // ALl the current state of the round.
    pub round_data: Vec<RoundData>,
    // The community cards.
    pub board: Vec<Card>,
}

impl GameState {
    pub fn new(stacks: Vec<i32>, big_blind: i32, small_blind: i32, dealer_idx: usize) -> Self {
        let num_players = stacks.len();
        let mut gs = GameState {
            num_players,
            stacks,
            big_blind,
            small_blind,
            player_active: PlayerBitSet::new(num_players),
            player_all_in: PlayerBitSet::default(),
            player_bet: vec![0; num_players],
            player_winnings: vec![0; num_players],
            dealer_idx,
            total_pot: 0,
            hands: vec![Hand::default(); num_players],
            round: Round::Starting,
            board: vec![],
            round_data: vec![],
        };
        gs.round_data.push(gs.new_round_data());
        gs
    }

    pub fn num_active_players(&self) -> usize {
        self.player_active.count()
    }

    pub fn num_active_players_in_round(&self) -> usize {
        self.current_round_data().num_active_players()
    }

    pub fn num_all_in_players(&self) -> usize {
        self.player_all_in.count()
    }

    pub fn is_complete(&self) -> bool {
        self.num_active_players() == 1 || self.round == Round::Complete
    }

    pub fn advance_round(&mut self) {
        match self.round {
            Round::Starting => self.advance_preflop(),
            Round::Complete => (),
            _ => self.advance_normal(),
        }
    }

    fn advance_preflop(&mut self) {
        self.round = Round::Preflop;
        self.round_data.push(self.new_round_data());
        self.do_bet(self.small_blind, true).unwrap();
        self.do_bet(self.big_blind, true).unwrap();
    }

    fn advance_normal(&mut self) {
        self.round = self.round.advance();
        self.round_data.push(self.new_round_data());
    }

    pub fn complete(&mut self) {
        self.round = Round::Complete;
        self.round_data.push(self.new_round_data());
    }

    fn new_round_data(&self) -> RoundData {
        let mut rd = RoundData {
            player_bet: vec![0; self.num_players],
            bet_count: vec![0; self.num_players],
            raise_count: vec![0; self.num_players],
            bet: 0,
            min_raise: self.big_blind,
            player_active: self.player_active,
            to_act_idx: self.dealer_idx,
        };

        rd.advance();

        rd
    }

    pub fn current_round_data(&self) -> &RoundData {
        self.round_data.last().unwrap()
    }

    pub fn mut_current_round_data(&mut self) -> &mut RoundData {
        self.round_data.last_mut().unwrap()
    }

    pub fn fold(&mut self) {
        // Which player is next to act
        let idx = self.current_round_data().to_act_idx;
        // We are going to change the current round since this player is out.
        self.mut_current_round_data().player_active.disable(idx);
        self.player_active.disable(idx);
        self.mut_current_round_data().advance();
    }

    pub fn do_bet(&mut self, ammount: i32, is_forced: bool) -> Result<i32, GameStateError> {
        // Which player is next to act
        let idx = self.current_round_data().to_act_idx;

        // This is the ammount extra that the player is putting into the round's betting
        // pot
        //
        // We need to validate it before making anychanges to the game state. This
        // allows us to return an error before getting into any bad gamestate.
        //
        // It also allows agents to be punished for putting in bad bet types.
        //
        // Make sure the bet is a correct ammount and if not
        // then cap it at the maximum the player can bet (Their stacks usually)
        let extra_ammount = if is_forced {
            self.validate_forced_bet_ammount(ammount)
        } else {
            self.validate_bet_ammount(ammount)?
        };
        let prev_bet = self.current_round_data().bet;
        // At this point we start making changes.
        // Take the money out.
        self.stacks[idx] -= extra_ammount;
        self.mut_current_round_data()
            .do_bet(extra_ammount, is_forced);

        self.player_bet[idx] += extra_ammount;

        self.total_pot += extra_ammount;

        let is_betting_reopened = prev_bet < self.current_round_data().bet;

        if is_betting_reopened {
            // This is a new max bet. We need to reset who can act in the round
            self.mut_current_round_data().player_active = self.player_active;
        }

        // If they put money into the pot then they are done this turn.
        if !is_forced {
            self.mut_current_round_data().player_active.disable(idx);
        }

        // We're out and can't continue
        if self.stacks[idx] == 0 {
            // Keep track of who's still active.
            self.player_active.disable(idx);
            // Keep track of going all in. We'll use that later on
            // to determine who's worth ranking.
            self.player_all_in.enable(idx);
            // It doesn' matter if this is a forced
            // bet if the player is out of money.
            self.mut_current_round_data().player_active.disable(idx);
        }

        // Advance the next to act.
        self.mut_current_round_data().advance();

        Ok(extra_ammount)
    }

    pub fn award(&mut self, player_idx: usize, ammount: i32) {
        self.stacks[player_idx] += ammount;
        self.player_winnings[player_idx] += ammount;
    }

    fn validate_forced_bet_ammount(&self, ammount: i32) -> i32 {
        // Which player is next to act
        let idx = self.current_round_data().to_act_idx;

        self.stacks[idx].min(ammount)
    }

    fn validate_bet_ammount(&self, ammount: i32) -> Result<i32, GameStateError> {
        // Which player is next to act
        let idx = self.current_round_data().to_act_idx;
        let current_round_data = self.current_round_data();

        if current_round_data.player_bet[idx] > ammount {
            // We've already bet more than this. No takes backs.
            Err(GameStateError::BetSizeDoesntCallSelf)
        } else {
            // How much extra are we putting in.
            let extra = ammount - current_round_data.player_bet[idx];

            // How much more are we putting in this time. Capped at the stack
            let capped_extra = self.stacks[idx].min(extra);
            // What our new player bet will be
            let capped_new_player_bet = current_round_data.player_bet[idx] + capped_extra;
            let current_bet = current_round_data.bet;
            // How much this is a raise.
            let raise = (capped_new_player_bet - current_bet).max(0);
            let is_all_in = capped_extra == self.stacks[idx];
            let is_raise = raise > 0;
            if capped_new_player_bet < current_round_data.bet && !is_all_in {
                // If we're not even calling and it's not an all in.
                Err(GameStateError::BetSizeDoesntCall)
            } else if is_raise && !is_all_in && raise < current_round_data.min_raise {
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
            .field("player_active", &self.player_active)
            .field("num_active_players", &self.num_active_players())
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
            .field("hands", &self.hands)
            .field("dealer_idx", &self.dealer_idx)
            .field("round", &self.round)
            .field("round_data", &self.round_data)
            .field("board", &self.board)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_around_call() {
        let stacks = vec![100; 4];
        let mut game_state = GameState::new(stacks, 10, 5, 1);
        game_state.advance_round();

        // 0 player, 1 dealer, 2 small blind, 3 big blind
        assert_eq!(0, game_state.current_round_data().to_act_idx);
        game_state.fold();
        game_state.fold();
        game_state.do_bet(10, false).unwrap();
        game_state.do_bet(10, false).unwrap();
        assert_eq!(0, game_state.current_round_data().num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // Flop
        game_state.advance_round();
        assert_eq!(2, game_state.current_round_data().to_act_idx);
        game_state.do_bet(0, false).unwrap();
        assert_eq!(3, game_state.current_round_data().to_act_idx);
        game_state.do_bet(0, false).unwrap();
        assert_eq!(0, game_state.current_round_data().num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // Turn
        game_state.advance_round();
        assert_eq!(2, game_state.current_round_data().to_act_idx);
        assert_eq!(2, game_state.current_round_data().num_active_players());
        game_state.do_bet(0, false).unwrap();
        game_state.do_bet(0, false).unwrap();
        assert_eq!(0, game_state.current_round_data().num_active_players());
        assert_eq!(2, game_state.num_active_players());

        // River
        game_state.advance_round();
        game_state.do_bet(0, false).unwrap();
        game_state.do_bet(0, false).unwrap();
        assert_eq!(0, game_state.current_round_data().num_active_players());
        assert_eq!(2, game_state.num_active_players());

        game_state.advance_round();
        assert_eq!(Round::Showdown, game_state.round);
    }

    #[test]
    fn test_cant_bet_less_0() {
        let stacks = vec![100; 5];
        let mut game_state = GameState::new(stacks, 2, 1, 0);
        game_state.advance_round();

        game_state.do_bet(33, false).unwrap();
        game_state.fold();
        let res = game_state.do_bet(20, false);

        assert_eq!(res.err(), Some(GameStateError::BetSizeDoesntCall));
    }

    #[test]
    fn test_cant_bet_less_with_all_in() {
        let stacks = vec![100, 50, 50, 100, 10];
        let mut game_state = GameState::new(stacks, 2, 1, 0);
        // Post blinds and setup next to act
        game_state.advance_round();

        // UTG raises to 10
        game_state.do_bet(10, false).unwrap();

        // UTG+1 has 10 remaining so betting 100 is overbetting
        // into an all in.
        game_state.do_bet(100, false).unwrap();

        // Dealer gets out of the way
        game_state.fold();

        // Small Blind raises to 20
        game_state.do_bet(20, false).unwrap();

        // Big Blind can't call the previous value.
        let res = game_state.do_bet(10, false);
        assert_eq!(res.err(), Some(GameStateError::BetSizeDoesntCall));
    }

    #[test]
    fn test_cant_under_minraise_bb() {
        let stacks = vec![500; 5];
        let mut game_state = GameState::new(stacks, 20, 10, 0);
        // Post blinds and setup next to act
        game_state.advance_round();

        // UTG raises to 33
        //
        // However the min raise is the big blind
        // so since the bb has already posted
        // we're not able to raise 13
        assert_eq!(
            Err(GameStateError::RaiseSizeTooSmall),
            game_state.do_bet(33, false)
        );
    }
}
