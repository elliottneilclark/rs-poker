use crate::core::{Card, Hand, PlayerBitSet, Rank};

use super::game_state::Round;

/// Represents an action that an agent can take in a game.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentAction {
    /// Folds the current hand.
    Fold,
    /// Bets the specified amount of money.
    Bet(f32),
}

#[derive(Debug, Clone, PartialEq)]
/// The game has started.
pub struct GameStartPayload {
    pub ante: f32,
    pub small_blind: f32,
    pub big_blind: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerSitPayload {
    pub idx: usize,
    pub player_stack: f32,
}

/// Each player is dealt a card. This is the payload for the event.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct DealStartingHandPayload {
    pub card: Card,
    pub idx: usize,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ForcedBetType {
    Ante,
    SmallBlind,
    BigBlind,
}

#[derive(Debug, Clone, PartialEq)]
/// A player tried to play an action and failed
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
/// A player tried to play an action and failed
pub struct PlayedActionPayload {
    // The tried Action
    pub action: AgentAction,

    pub idx: usize,
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

#[derive(Debug, Clone, PartialEq)]
/// A player tried to play an action and failed
pub struct FailedActionPayload {
    // The tried Action
    pub action: AgentAction,
    // The result action
    pub result: PlayedActionPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AwardPayload {
    pub total_pot: f32,
    pub award_amount: f32,
    pub rank: Option<Rank>,
    pub hand: Option<Hand>,
    pub idx: usize,
}

/// Represents an action that can happen in a game.
#[derive(Debug, Clone, PartialEq)]
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
    use super::*;

    #[test]
    fn test_bet() {
        let a = AgentAction::Bet(100.0);
        assert_eq!(AgentAction::Bet(100.0), a);
    }
}
