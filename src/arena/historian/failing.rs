use tracing::warn;

use crate::arena::{GameState, Historian};

/// A historian that will always fail to record an action
/// and will return an error.
///
/// This historian is useful for testing the behavior of the simulation
pub struct FailingHistorian;

impl Historian for FailingHistorian {
    fn record_action(
        &mut self,
        _id: u128,
        _game_state: &GameState,
        _action: crate::arena::action::Action,
    ) -> Result<(), crate::arena::historian::HistorianError> {
        warn!("FailingHistorian intentionally returning error");
        Err(crate::arena::historian::HistorianError::UnableToRecordAction)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{HoldemSimulationBuilder, agent::CallingAgent};

    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    #[should_panic]
    fn test_panic_fail_historian() {
        let historian = Box::new(FailingHistorian);

        let stacks = vec![100.0; 3];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(CallingAgent::new("CallingAgent-fail-0")),
                Box::new(CallingAgent::new("CallingAgent-fail-1")),
                Box::new(CallingAgent::new("CallingAgent-fail-2")),
            ])
            .panic_on_historian_error(true)
            .historians(vec![historian])
            .build()
            .unwrap();

        // This should panic since panic_on_historian_error is set to true
        // and the historian will always fail to record an action
        sim.run(&mut rng);
    }
}
