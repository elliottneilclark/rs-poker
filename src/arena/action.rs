use crate::core::Card;

use super::game_state::Round;

/// Represents an action that an agent can take in a game.
#[derive(Debug, Clone, PartialEq, Copy, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentAction {
    /// Folds the current hand.
    Fold,
    /// Bets the specified amount of money.
    Bet(i32),
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
/// The game has started.
pub struct GameStartPayload {
    pub small_blind: i32,
    pub big_blind: i32,
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
pub struct PlayerSitPayload {
    pub idx: usize,
    pub player_stack: i32,
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
/// Each player is dealt two cards.
pub struct DealStartingHandPayload {
    pub card_one: Card,
    pub card_two: Card,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
/// A player tried to play an action and failed
pub struct ForcedBetPayload {
    /// A bet that the player is forced to make
    /// The ammount is the forced ammount, not the final
    /// amount which could be lower if that puts the player all in.
    pub bet: i32,
    pub idx: usize,
    pub player_stack: i32,
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
/// A player tried to play an action and failed
pub struct PlayedActionPayload {
    // The tried Action
    pub action: AgentAction,
    pub idx: usize,
    pub player_stack: i32,
}

#[derive(Debug, Clone, PartialEq, Copy, Hash)]
/// A player tried to play an action and failed
pub struct FailedActionPayload {
    // The tried Action
    pub action: AgentAction,
    // The result action
    pub result_action: AgentAction,
    pub idx: usize,
    pub player_stack: i32,
}

/// Represents an action that can happen in a game.
#[derive(Debug, Clone, PartialEq, Copy, Hash)]
pub enum Action {
    GameStart(GameStartPayload),
    PlayerSit(PlayerSitPayload),
    DealStartingHand(DealStartingHandPayload),
    /// The round has advanced.
    RoundAdvance(Round),
    ForcedBet(ForcedBetPayload),
    /// A player has played an action.
    PlayedAction(PlayedActionPayload),
    /// The player tried and failed to take some action.
    FailedAction(FailedActionPayload),
    /// A community card has been dealt.
    DealCommunity(Card),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bet() {
        let a = AgentAction::Bet(100);
        assert_eq!(AgentAction::Bet(100), a);
    }
}
