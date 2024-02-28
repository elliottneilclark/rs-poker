use std::{cell::RefCell, rc::Rc};

use crate::arena::action::Action;

use super::Historian;

/// VecHistorian is a historian that will
/// append each action to a vector.
pub struct VecHistorian {
    actions: Rc<RefCell<Vec<Action>>>,
}

impl VecHistorian {
    pub fn new(actions: Rc<RefCell<Vec<Action>>>) -> Self {
        Self { actions }
    }
}

impl Historian for VecHistorian {
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        _game_state: &crate::arena::GameState,
        action: Action,
    ) {
        self.actions.borrow_mut().push(action)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{agent::RandomAgent, Agent, GameState, HoldemSimulationBuilder};

    use super::*;

    #[test]
    fn test_vec_historian() {
        let actions = Rc::new(RefCell::new(Vec::new()));
        let hist = Box::new(VecHistorian::new(actions.clone()));

        let stacks = vec![100.0; 5];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
        ];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run();

        assert!(actions.borrow().len() > 10);
    }
}
