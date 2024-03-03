use crate::arena::{action::Action, GameState};

use super::{Historian, HistorianError};

/// This `Agent` is an implmentation that returns
/// random actions. However, it also takes in a function
/// that is called when an action is received. This is
/// useful for testing and debugging.
#[derive(Debug, Clone)]
pub struct FnHistorian<F> {
    func: F,
}

impl<F: Fn(&uuid::Uuid, &GameState, Action) -> Result<(), HistorianError>> FnHistorian<F> {
    /// Create a new `FnHistorian` with the provided function
    /// that will be called when an action is received on a simulation.
    pub fn new(f: F) -> Self {
        Self { func: f }
    }
}

impl<F: Clone + Fn(&uuid::Uuid, &GameState, Action) -> Result<(), HistorianError>> Historian
    for FnHistorian<F>
{
    fn record_action(
        &mut self,
        id: &uuid::Uuid,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        // Call the function with the action that was received
        (self.func)(id, game_state, action)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::arena::{agent::RandomAgent, game_state::Round, Agent, HoldemSimulationBuilder};

    use super::*;

    #[test]
    fn test_can_record_actions_with_agents() {
        let last_action: Rc<RefCell<Option<Action>>> = Rc::new(RefCell::new(None));
        let count = Rc::new(RefCell::new(0));

        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|_| Box::<RandomAgent>::default() as Box<dyn Agent>)
            .collect();
        let game_state = GameState::new(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);

        let borrow_count = count.clone();
        let borrow_last_action = last_action.clone();

        let historian = Box::new(FnHistorian::new(move |_id, _game_state, action| {
            *borrow_count.borrow_mut() += 1;
            *borrow_last_action.borrow_mut() = Some(action);
            Ok(())
        }));

        let mut sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .historians(vec![historian])
            .build()
            .unwrap();

        sim.run();

        assert_ne!(0, count.take());

        let act = last_action.take();

        assert!(act.is_some());

        assert_eq!(Some(Action::RoundAdvance(Round::Complete)), act);
    }

    #[test]
    fn test_fn_historian_can_withstand_error() {
        // A test that adds a historian that always returns an error
        // This shows that the historian will be dropped from the simulation
        // if it returns an error but the simulation will continue to run.

        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|_| Box::<RandomAgent>::default() as Box<dyn Agent>)
            .collect();

        let game_state = GameState::new(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        let historian = Box::new(FnHistorian::new(|_, _, _| {
            Err(HistorianError::UnableToRecordAction)
        }));

        HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .historians(vec![historian])
            .build()
            .unwrap()
            .run();
    }
}
