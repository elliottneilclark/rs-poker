use crate::core::{Card, Hand, PlayerBitSet, Rank};

use super::game_state::Round;

/// Represents an action that an agent can take in a game.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentAction {
    /// Folds the current hand.
    Fold,
    /// Matches the current bet.
    Call,
    /// Bets the specified amount of money.
    Bet(f32),
    /// Go all-in
    AllIn,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// The game has started.
pub struct GameStartPayload {
    pub ante: f32,
    pub small_blind: f32,
    pub big_blind: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayerSitPayload {
    pub idx: usize,
    pub player_stack: f32,
    /// Optional agent name reported by the simulator so historians can preserve it.
    pub name: Option<String>,
}

/// Each player is dealt a card. This is the payload for the event.
#[derive(Debug, Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DealStartingHandPayload {
    pub card: Card,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForcedBetType {
    Ante,
    SmallBlind,
    BigBlind,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForcedBetPayload {
    /// A bet that the player is forced to make
    /// The amount is the forced amount, not the final
    /// amount which could be lower if that puts the player all in.
    pub bet: f32,
    pub player_stack: f32,
    pub idx: usize,
    pub forced_bet_type: ForcedBetType,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayedActionPayload {
    // The tried Action
    pub action: AgentAction,

    pub idx: usize,
    pub round: Round,
    pub player_stack: f32,

    pub starting_pot: f32,
    pub final_pot: f32,

    pub starting_bet: f32,
    pub final_bet: f32,

    pub starting_min_raise: f32,
    pub final_min_raise: f32,

    pub starting_player_bet: f32,
    pub final_player_bet: f32,

    pub players_active: PlayerBitSet,
    pub players_all_in: PlayerBitSet,
}

impl PlayedActionPayload {
    pub fn raise_amount(&self) -> f32 {
        self.final_bet - self.starting_bet
    }
}

/// A player tried to play an action and failed
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FailedActionPayload {
    // The tried Action
    pub action: AgentAction,
    // The result action
    pub result: PlayedActionPayload,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AwardPayload {
    pub total_pot: f32,
    pub award_amount: f32,
    pub rank: Option<Rank>,
    pub hand: Option<Hand>,
    pub idx: usize,
}

/// Represents an action that can happen in a game.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    GameStart(GameStartPayload),
    PlayerSit(PlayerSitPayload),
    DealStartingHand(DealStartingHandPayload),
    /// The round has advanced.
    RoundAdvance(Round),
    /// A player has played an action.
    PlayedAction(PlayedActionPayload),
    /// The player tried and failed to take some action.
    /// If the action failed then there is no PlayedAction event coming.
    ///
    /// Players can fail to fold when there's no money being wagered.
    /// Players can fail to bet when they bet an illegal amount.
    FailedAction(FailedActionPayload),

    /// A player/agent was forced to make a bet.
    ForcedBet(ForcedBetPayload),
    /// A community card has been dealt.
    DealCommunity(Card),
    /// There was some pot given to a player
    Award(AwardPayload),
}

#[cfg(test)]
mod tests {
    use crate::core::PlayerBitSet;

    use super::*;

    #[test]
    fn test_bet() {
        let a = AgentAction::Bet(100.0);
        assert_eq!(AgentAction::Bet(100.0), a);
    }

    /// Verifies raise_amount correctly calculates the increase from starting to final bet.
    #[test]
    fn test_raise_amount_calculation() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(100.0),
            idx: 0,
            round: Round::Preflop,
            player_stack: 500.0,
            starting_pot: 15.0,
            final_pot: 115.0,
            starting_bet: 10.0,
            final_bet: 30.0, // Raise from 10 to 30
            starting_min_raise: 10.0,
            final_min_raise: 20.0,
            starting_player_bet: 0.0,
            final_player_bet: 30.0,
            players_active: PlayerBitSet::new(2),
            players_all_in: PlayerBitSet::default(),
        };

        // raise_amount = final_bet - starting_bet = 30 - 10 = 20
        assert_eq!(payload.raise_amount(), 20.0);
    }

    /// Verifies raise_amount with a raise from 25 to 75 (50 raise amount).
    #[test]
    fn test_raise_amount_different_values() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(50.0),
            idx: 0,
            round: Round::Flop,
            player_stack: 200.0,
            starting_pot: 50.0,
            final_pot: 100.0,
            starting_bet: 25.0,
            final_bet: 75.0,
            starting_min_raise: 25.0,
            final_min_raise: 50.0,
            starting_player_bet: 0.0,
            final_player_bet: 75.0,
            players_active: PlayerBitSet::new(3),
            players_all_in: PlayerBitSet::default(),
        };

        assert_eq!(payload.raise_amount(), 50.0);
    }

    /// Test raise_amount with zero raise (check scenario).
    #[test]
    fn test_raise_amount_no_raise() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(10.0),
            idx: 1,
            round: Round::Preflop,
            player_stack: 100.0,
            starting_pot: 15.0,
            final_pot: 25.0,
            starting_bet: 10.0,
            final_bet: 10.0, // No raise - just called
            starting_min_raise: 10.0,
            final_min_raise: 10.0,
            starting_player_bet: 0.0,
            final_player_bet: 10.0,
            players_active: PlayerBitSet::new(2),
            players_all_in: PlayerBitSet::default(),
        };

        // No raise was made (just called), so raise_amount is 0
        assert_eq!(payload.raise_amount(), 0.0);
    }
}
