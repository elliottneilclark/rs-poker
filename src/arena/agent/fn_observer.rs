use crate::arena::{
    action::{Action, AgentAction},
    game_state::GameState,
};

use super::{random::RandomAgent, Agent};

/// This `Agent` is an implmentation that returns
/// random actions. However, it also takes in a function
/// that is called when an action is received. This is
/// useful for testing and debugging.
#[derive(Debug, Clone)]
pub struct FnObserverRandomAgent<F> {
    func: F,
    random_agent: RandomAgent,
}

impl<F: Fn(&Action)> FnObserverRandomAgent<F> {
    pub fn new(f: F) -> Self {
        Self {
            func: f,
            random_agent: RandomAgent::default(),
        }
    }
}

impl<F: Clone + Fn(&Action)> Agent for FnObserverRandomAgent<F> {
    fn act(&mut self, game_state: &GameState) -> AgentAction {
        // Delegate to the random agent
        self.random_agent.act(game_state)
    }

    fn action_received(&mut self, _game_state: &GameState, action: &Action) {
        // Call the function with the action that was received
        (self.func)(action);
    }
}

#[cfg(test)]
mod tests {

    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::arena::HoldemSimulationBuilder;

    #[test]
    fn test_observer_agent() {
        let game_state = GameState::new(vec![100.0, 100.0], 10.0, 5.0, 0);

        let count = Rc::new(RefCell::new(0));
        let last_action: Rc<RefCell<Option<Action>>> = Rc::new(RefCell::new(None));

        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|_| {
                let local_count = count.clone();
                let local_action = last_action.clone();

                let a = FnObserverRandomAgent::new(move |action: &Action| {
                    *local_count.borrow_mut() += 1;
                    *local_action.borrow_mut() = Some(action.clone());
                });
                Box::new(a) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .build()
            .unwrap();

        sim.run();

        assert_ne!(0, count.take());

        let act = last_action.take();

        assert!(act.is_some());
    }
}
