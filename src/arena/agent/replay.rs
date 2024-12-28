use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

/// A replay agent that will replay a sequence of actions
/// from a vector. It consumes the vector making it fast but
/// hard to reuse or introspect what actions were taken.
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

/// A replay agent that will replay a sequence of actions from a slice.
#[derive(Debug, Clone)]
pub struct SliceReplayAgent<'a> {
    actions: &'a [AgentAction],
    idx: usize,
    default: AgentAction,
}

impl<'a> SliceReplayAgent<'a> {
    pub fn new(actions: &'a [AgentAction]) -> Self {
        Self {
            actions,
            idx: 0,
            default: AgentAction::Fold,
        }
    }
}

impl Agent for VecReplayAgent {
    fn act(self: &mut VecReplayAgent, _id: &uuid::Uuid, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        self.actions
            .get(idx)
            .map_or_else(|| self.default.clone(), |a| a.clone())
    }
}

impl<'a> Agent for SliceReplayAgent<'a> {
    fn act(
        self: &mut SliceReplayAgent<'a>,
        _id: &uuid::Uuid,
        _game_state: &GameState,
    ) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        self.actions
            .get(idx)
            .map_or_else(|| self.default.clone(), |a| a.clone())
    }
}

#[cfg(test)]
mod tests {

    use rand::{rngs::StdRng, SeedableRng};

    use crate::arena::{
        action::AgentAction,
        agent::VecReplayAgent,
        test_util::{assert_valid_game_state, assert_valid_round_data},
        Agent, GameState, HoldemSimulation, RngHoldemSimulationBuilder,
    };

    #[test_log::test]
    fn test_all_in_for_less() {
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(690.0),
        ]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(690.0),
        ]));
        let agent_three = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(90.0),
        ]));
        let agent_four = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(10.0),
            AgentAction::Fold,
        ]));

        let stacks = vec![700.0, 900.0, 100.0, 800.0];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
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

    #[test]
    fn test_cant_bet_after_folds() {
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![]));
        let agent_three =
            Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Bet(100.0)]));

        let stacks = vec![100.0, 100.0, 100.0];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three];
        let rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run();

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_another_three_player() {
        let sb = 3.0;
        let bb = 3.0;

        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(bb),
            AgentAction::Bet(bb),
        ]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(bb),
            AgentAction::Bet(bb),
        ]));
        let agent_three = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Fold]));

        let stacks = vec![bb + 5.906776e-3, bb + 5.906776e-39, bb];
        let game_state = GameState::new_starting(stacks, bb, sb, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three];
        let rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
            .rng(rng)
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run();

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test_log::test]
    fn test_from_fuzz_early_all_in() {
        // This test was discoverd by fuzzing.
        let agent_zero = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Fold]));
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Fold]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Fold]));
        let agent_three =
            Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Bet(5.0)]));
        let agent_four =
            Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Bet(5.0)]));
        let agent_five = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259.0),
            AgentAction::Fold,
        ]));

        let stacks = vec![1000.0, 100.0, 1000.0, 5.0, 5.0, 1000.0];
        let game_state = GameState::new_starting(stacks, 114.0, 96.0, 0.0, 210439175936 % 5);
        let agents: Vec<Box<dyn Agent>> = vec![
            agent_zero,
            agent_one,
            agent_two,
            agent_three,
            agent_four,
            agent_five,
        ];
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

    #[test_log::test]
    fn test_from_fuzz() {
        // This test was discoverd by fuzzing.
        //
        // Previously it would fail as the last two agents in
        // a round both fold leaving orphaned money in the pot.
        let agent_one = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![]));
        let agent_two = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259.0),
            AgentAction::Bet(16711936.0),
        ]));
        let agent_three = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Fold,
        ]));
        let agent_four =
            Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![AgentAction::Bet(57828.0)]));
        let agent_five = Box::<VecReplayAgent>::new(VecReplayAgent::new(vec![
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Fold,
        ]));

        let stacks = vec![22784.0, 260.0, 65471.0, 255.0, 65471.0];
        let game_state = GameState::new_starting(stacks, 114.0, 96.0, 0.0, 210439175936 % 5);
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
