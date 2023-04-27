use crate::core::Card;

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AgentAction {
    Fold,
    Bet(i32),
}

pub enum Action {
    GameStart,
    DealStartingHand(Card, Card),
    RoundAdvance,
    PlayedAction(AgentAction),
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
