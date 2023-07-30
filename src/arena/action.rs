use crate::core::Card;

/// Represents an action that an agent can take in a game.
#[derive(Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentAction {
    /// Folds the current hand.
    Fold,
    /// Bets the specified amount of money.
    Bet(i32),
}

/// Represents an action that can happen in a game.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Action {
    /// The game has started.
    GameStart,
    /// Each player is dealt two cards.
    DealStartingHand(Card, Card),
    /// The round has advanced.
    RoundAdvance,
    /// A bet that the player is forced to make
    /// The ammount is the forced ammount, not the final
    /// amount which could be lower if that puts the player all in.
    ForcedBet(i32),
    /// A player has played an action.
    PlayedAction(AgentAction),
    /// A player tried to play an action and failed
    FailedAction(
        // The tried Action
        AgentAction,
        // The result action
        AgentAction,
    ),
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
