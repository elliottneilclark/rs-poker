use crate::core::{Card, Hand, Rank};

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
    pub small_blind: f32,
    pub big_blind: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerSitPayload {
    pub idx: usize,
    pub player_stack: f32,
}

#[derive(Debug, Clone, PartialEq, Hash)]
/// Each player is dealt a card. This is the payload for the event.
pub struct DealStartingHandPayload {
    pub card: Card,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// A player tried to play an action and failed
pub struct ForcedBetPayload {
    /// A bet that the player is forced to make
    /// The ammount is the forced ammount, not the final
    /// amount which could be lower if that puts the player all in.
    pub bet: f32,
    pub player_stack: f32,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// A player tried to play an action and failed
pub struct PlayedActionPayload {
    // The tried Action
    pub action: AgentAction,
    pub idx: usize,
    pub player_stack: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// A player tried to play an action and failed
pub struct FailedActionPayload {
    // The tried Action
    pub action: AgentAction,
    // The result action
    pub result_action: AgentAction,
    pub player_stack: f32,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AwardPayload {
    pub total_pot: f32,
    pub award_ammount: f32,
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
    /// Players can fail to bet when they bet an illegal ammount.
    FailedAction(FailedActionPayload),
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
