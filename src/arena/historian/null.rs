use super::Historian;

/// A no-op historian that discards all recorded actions.
///
/// Useful when you need a historian but don't care about the history.
pub struct NullHistorian;

impl Historian for NullHistorian {
    fn record_action(
        &mut self,
        _id: u128,
        _game_state: &crate::arena::GameState,
        _action: crate::arena::action::Action,
    ) -> Result<(), super::HistorianError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameState;
    use crate::arena::action::{Action, GameStartPayload};

    #[test]
    fn test_null_historian_accepts_actions() {
        let mut historian = NullHistorian;
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
        let action = Action::GameStart(GameStartPayload {
            ante: 0.0,
            small_blind: 5.0,
            big_blind: 10.0,
        });

        let result = historian.record_action(123, &game_state, action);
        assert!(result.is_ok());
    }
}
