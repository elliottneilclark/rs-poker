use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

#[derive(Debug, Clone)]
pub struct VecReplayAgent {
    actions: Vec<AgentAction>,
    idx: usize,
    default: AgentAction,
}

impl VecReplayAgent {
    pub fn new(actions: Vec<AgentAction>) -> Self {
        Self {
            actions,
            idx: 0,
            default: AgentAction::Fold,
        }
    }
}

#[derive(Clone)]
pub struct SliceReplayAgent<'a> {
    actions: &'a [AgentAction],
    idx: usize,
    default: AgentAction,
}

impl Agent for VecReplayAgent {
    fn act(self: &mut VecReplayAgent, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        *self.actions.get(idx).unwrap_or(&self.default)
    }
}

impl<'a> Agent for SliceReplayAgent<'a> {
    fn act(self: &mut SliceReplayAgent<'a>, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        *self.actions.get(idx).unwrap_or(&self.default)
    }
}

#[cfg(test)]
mod tests {

    use rand::{rngs::StdRng, SeedableRng};

    use crate::arena::{
        action::AgentAction, agent::VecReplayAgent, test_util::assert_valid_game_state, Agent,
        GameState, HoldemSimulation, RngHoldemSimulationBuilder,
    };

    #[test_log::test]
    fn test_all_in_for_less() {
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10),
            AgentAction::Bet(0),
            AgentAction::Bet(0),
            AgentAction::Bet(690),
        ]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10),
            AgentAction::Bet(0),
            AgentAction::Bet(0),
            AgentAction::Bet(690),
        ]));
        let agent_three = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10),
            AgentAction::Bet(0),
            AgentAction::Bet(0),
            AgentAction::Bet(90),
        ]));
        let agent_four = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10),
            AgentAction::Fold,
        ]));

        let stacks = vec![700, 900, 100, 800];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three, agent_four];
        let rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run();

        assert_valid_game_state(&sim.game_state);
    }

    #[test_log::test]
    fn test_from_fuzz() {
        // This test was discoverd by fuzzing.
        //
        // Previously it would fail as the last two agents in
        // a round both fold leaving orphaned money in the pot.
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259),
            AgentAction::Bet(16711936),
        ]));
        let agent_three = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259),
            AgentAction::Bet(259),
            AgentAction::Bet(259),
            AgentAction::Fold,
        ]));
        let agent_four =
            Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Bet(57828)]));
        let agent_five = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259),
            AgentAction::Bet(259),
            AgentAction::Bet(259),
            AgentAction::Fold,
        ]));

        let stacks = vec![22784, 260, 65471, 255, 65471];
        let game_state = GameState::new(stacks, 114, 96, 210439175936 % 5);
        let agents: Vec<Box<dyn Agent>> =
            vec![agent_one, agent_two, agent_three, agent_four, agent_five];
        let rng = StdRng::seed_from_u64(0);

        let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run();

        assert_valid_game_state(&sim.game_state);
    }
}
