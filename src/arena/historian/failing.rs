use crate::arena::{GameState, Historian};

/// A historian that will always fail to record an action
/// and will return an error.
///
/// This historian is useful for testing the behavior of the simulation
pub struct FailingHistorian;

impl Historian for FailingHistorian {
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        _game_state: &GameState,
        _action: crate::arena::action::Action,
    ) -> Result<(), crate::arena::historian::HistorianError> {
        Err(crate::arena::historian::HistorianError::UnableToRecordAction)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{agent::CallingAgent, HoldemSimulationBuilder};

    use super::*;

    #[test]
    #[should_panic]
    fn test_panic_fail_historian() {
        let historian = Box::new(FailingHistorian);

        let stacks = vec![100.0; 3];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
            ])
            .panic_on_historian_error(true)
            .historians(vec![historian])
            .build()
            .unwrap();

        // This should panic since panic_on_historian_error is set to true
        // and the historian will always fail to record an action
        sim.run()
    }
}
