use tracing::{debug, instrument, trace};

use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

/// A replay agent that will replay a sequence of actions
/// from a vector. It consumes the vector making it fast but
/// hard to reuse or introspect what actions were taken.
#[derive(Debug, Clone)]
pub struct VecReplayAgent {
    name: String,
    actions: Vec<AgentAction>,
    idx: usize,
    default: AgentAction,
}

impl VecReplayAgent {
    pub fn new(name: impl Into<String>, actions: Vec<AgentAction>) -> Self {
        Self::new_with_default(name, actions, AgentAction::Fold)
    }

    pub fn new_with_default(
        name: impl Into<String>,
        actions: Vec<AgentAction>,
        default: AgentAction,
    ) -> Self {
        Self {
            name: name.into(),
            actions,
            idx: 0,
            default,
        }
    }
}

/// A replay agent that will replay a sequence of actions from a slice.
#[derive(Debug, Clone)]
pub struct SliceReplayAgent<'a> {
    name: String,
    actions: &'a [AgentAction],
    idx: usize,
    default: AgentAction,
}

impl<'a> SliceReplayAgent<'a> {
    pub fn new(name: impl Into<String>, actions: &'a [AgentAction]) -> Self {
        Self::new_with_default(name, actions, AgentAction::Fold)
    }

    pub fn new_with_default(
        name: impl Into<String>,
        actions: &'a [AgentAction],
        default: AgentAction,
    ) -> Self {
        Self {
            name: name.into(),
            actions,
            idx: 0,
            default,
        }
    }
}

impl Agent for VecReplayAgent {
    #[instrument(level = "trace", skip(self, _game_state), fields(agent_name = %self.name))]
    fn act(self: &mut VecReplayAgent, _id: u128, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        self.actions.get(idx).map_or_else(
            || {
                debug!(
                    idx,
                    actions_len = self.actions.len(),
                    ?self.default,
                    "VecReplayAgent exhausted actions, using default"
                );
                self.default.clone()
            },
            |a| {
                trace!(idx, ?a, "VecReplayAgent replaying action");
                a.clone()
            },
        )
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl<'a> Agent for SliceReplayAgent<'a> {
    #[instrument(level = "trace", skip(self, _game_state), fields(agent_name = %self.name))]
    fn act(self: &mut SliceReplayAgent<'a>, _id: u128, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        self.actions.get(idx).map_or_else(
            || {
                debug!(
                    idx,
                    actions_len = self.actions.len(),
                    ?self.default,
                    "SliceReplayAgent exhausted actions, using default"
                );
                self.default.clone()
            },
            |a| {
                trace!(idx, ?a, "SliceReplayAgent replaying action");
                a.clone()
            },
        )
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {

    use std::sync::atomic::{AtomicUsize, Ordering};

    use rand::{SeedableRng, rngs::StdRng};

    use crate::arena::{
        Agent, GameStateBuilder, HoldemSimulation, HoldemSimulationBuilder,
        action::AgentAction,
        agent::VecReplayAgent,
        test_util::{assert_valid_game_state, assert_valid_round_data},
    };

    fn boxed_vec_agent(actions: Vec<AgentAction>) -> Box<VecReplayAgent> {
        boxed_vec_agent_with_default(actions, AgentAction::Fold)
    }

    fn boxed_vec_agent_with_default(
        actions: Vec<AgentAction>,
        default: AgentAction,
    ) -> Box<VecReplayAgent> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let name = format!(
            "vec-replay-agent-{}",
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        Box::new(VecReplayAgent::new_with_default(name, actions, default))
    }

    #[test]
    fn test_all_in_for_less() {
        let agent_one = boxed_vec_agent(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(690.0),
        ]);
        let agent_two = boxed_vec_agent(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(690.0),
        ]);
        let agent_three = boxed_vec_agent(vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(0.0),
            AgentAction::Bet(90.0),
        ]);
        let agent_four = boxed_vec_agent(vec![AgentAction::Bet(10.0), AgentAction::Fold]);

        let stacks = vec![700.0, 900.0, 100.0, 800.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three, agent_four];
        let mut rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run(&mut rng);

        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_cant_bet_after_folds() {
        let agent_one = boxed_vec_agent(vec![]);
        let agent_two = boxed_vec_agent(vec![]);
        let agent_three = boxed_vec_agent(vec![AgentAction::Bet(100.0)]);

        let stacks = vec![100.0, 100.0, 100.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three];
        let mut rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_another_three_player() {
        let sb = 3.0;
        let bb = 3.0;

        let agent_one = boxed_vec_agent(vec![AgentAction::Bet(bb), AgentAction::Bet(bb)]);
        let agent_two = boxed_vec_agent(vec![AgentAction::Bet(bb), AgentAction::Bet(bb)]);
        let agent_three = boxed_vec_agent(vec![AgentAction::Fold]);

        let stacks = vec![bb + 5.906776e-3, bb + 5.906776e-39, bb];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(bb, sb)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_one, agent_two, agent_three];
        let mut rng = StdRng::seed_from_u64(421);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run(&mut rng);

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_from_fuzz_early_all_in() {
        // This test was discoverd by fuzzing.
        let agent_zero = boxed_vec_agent(vec![AgentAction::Fold]);
        let agent_one = boxed_vec_agent(vec![AgentAction::Fold]);
        let agent_two = boxed_vec_agent(vec![AgentAction::Fold]);
        let agent_three = boxed_vec_agent(vec![AgentAction::Bet(5.0)]);
        let agent_four = boxed_vec_agent(vec![AgentAction::Bet(5.0)]);
        let agent_five = boxed_vec_agent(vec![AgentAction::Bet(259.0), AgentAction::Fold]);

        let stacks = vec![1000.0, 100.0, 1000.0, 5.0, 5.0, 1000.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(114.0, 96.0)
            .dealer_idx(210439175936 % 5)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![
            agent_zero,
            agent_one,
            agent_two,
            agent_three,
            agent_four,
            agent_five,
        ];
        let mut rng = StdRng::seed_from_u64(0);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_from_fuzz() {
        // This test was discoverd by fuzzing.
        //
        // Previously it would fail as the last two agents in
        // a round both fold leaving orphaned money in the pot.
        let agent_one = boxed_vec_agent(vec![]);
        let agent_two =
            boxed_vec_agent(vec![AgentAction::Bet(259.0), AgentAction::Bet(16711936.0)]);
        let agent_three = boxed_vec_agent(vec![
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Fold,
        ]);
        let agent_four = boxed_vec_agent(vec![AgentAction::Bet(57828.0)]);
        let agent_five = boxed_vec_agent(vec![
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Bet(259.0),
            AgentAction::Fold,
        ]);

        let stacks = vec![22784.0, 260.0, 65471.0, 255.0, 65471.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(114.0, 96.0)
            .dealer_idx(210439175936 % 5)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> =
            vec![agent_one, agent_two, agent_three, agent_four, agent_five];
        let mut rng = StdRng::seed_from_u64(0);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();
        sim.run(&mut rng);

        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_another_from_fuzz() {
        let agent_zero = boxed_vec_agent(vec![
            AgentAction::Fold,
            AgentAction::Fold,
            AgentAction::Fold,
            AgentAction::Fold,
            AgentAction::Fold,
            AgentAction::Fold,
            AgentAction::Fold,
        ]);
        let agent_one = boxed_vec_agent(vec![]);
        let stacks = vec![2.8460483e26, 53477376.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(8365616.5)
            .small_blind(0.0)
            .dealer_idx(1)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_zero, agent_one];
        let mut rng = StdRng::seed_from_u64(0);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_call_with_fold() {
        let agent_zero = boxed_vec_agent(vec![AgentAction::Call]);
        let agent_one = boxed_vec_agent(vec![
            AgentAction::Call,
            AgentAction::Fold,
            AgentAction::Fold,
        ]);
        let agent_two = boxed_vec_agent(vec![AgentAction::Call]);
        let agent_three = boxed_vec_agent(vec![AgentAction::Call, AgentAction::Call]);

        let stacks = vec![50000.0, 50000.0, 50000.0, 50000.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(50.0)
            .small_blind(3.59e-43)
            .dealer_idx(1)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_zero, agent_one, agent_two, agent_three];
        let mut rng = StdRng::seed_from_u64(0);

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_valid_game_state(&sim.game_state);
    }

    /// Verifies that VecReplayAgent::name() returns the name provided at construction.
    #[test]
    fn test_vec_replay_agent_name() {
        let agent = VecReplayAgent::new("TestAgentName", vec![AgentAction::Fold]);
        assert_eq!(agent.name(), "TestAgentName");
        assert!(!agent.name().is_empty());
        assert_ne!(agent.name(), "xyzzy");
    }

    /// Verifies that SliceReplayAgent::name() returns the name provided at construction.
    #[test]
    fn test_slice_replay_agent_name() {
        use super::SliceReplayAgent;
        let actions = vec![AgentAction::Fold, AgentAction::Bet(10.0)];
        let agent = SliceReplayAgent::new("SliceAgentName", &actions);
        assert_eq!(agent.name(), "SliceAgentName");
        assert!(!agent.name().is_empty());
        assert_ne!(agent.name(), "xyzzy");
    }

    /// Verifies that SliceReplayAgent::act() returns actions in sequence.
    #[test]
    fn test_slice_replay_agent_index_increment() {
        use super::SliceReplayAgent;
        let actions = vec![
            AgentAction::Bet(10.0),
            AgentAction::Bet(20.0),
            AgentAction::Bet(30.0),
        ];
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let mut agent = SliceReplayAgent::new("TestAgent", &actions);

        // First call should return first action
        let action1 = agent.act(0, &game_state);
        assert_eq!(action1, AgentAction::Bet(10.0));

        // Second call should return second action (idx incremented by 1, not multiplied/subtracted)
        let action2 = agent.act(0, &game_state);
        assert_eq!(action2, AgentAction::Bet(20.0));

        // Third call should return third action
        let action3 = agent.act(0, &game_state);
        assert_eq!(action3, AgentAction::Bet(30.0));

        // Fourth call should return default (exhausted)
        let action4 = agent.act(0, &game_state);
        assert_eq!(action4, AgentAction::Fold);
    }

    /// Test that all-in players should not have any actions on subsequent streets.
    /// When both players go all-in preflop, the simulation should not ask them
    /// to act on flop/turn/river, and no Check actions should be recorded.
    #[test]
    #[cfg(feature = "open-hand-history")]
    fn test_all_in_players_no_actions_on_subsequent_streets() {
        use crate::arena::historian::{self, OpenHandHistoryVecHistorian, VecHistorian};
        use crate::open_hand_history::{
            assert_open_hand_history_matches_game_state, assert_valid_open_hand_history,
        };

        // Reproduce the fuzzer crash scenario:
        // - 2 players with 3.6171875 stack each
        // - SB: 0.052481495, BB: 2.0042896
        // - Preflop: SB calls, BB raises all-in, SB calls all-in
        // - Both players are all-in after preflop, no actions should happen after

        // Player 0 (SB/dealer in heads-up):
        // - Posts SB (forced)
        // - Calls to BB level (needs ~1.95 more)
        // - Calls BB's all-in raise (needs ~1.61 more)
        let agent_zero = boxed_vec_agent_with_default(
            vec![AgentAction::Call, AgentAction::Call],
            AgentAction::Fold, // Should never be used since player is all-in
        );

        // Player 1 (BB):
        // - Posts BB (forced)
        // - Raises all-in with remaining stack (~1.61)
        let agent_one = boxed_vec_agent_with_default(
            vec![AgentAction::AllIn],
            AgentAction::Fold, // Should never be used since player is all-in
        );

        let stacks = vec![3.6171875, 3.6171875];
        let sb = 0.052481495;
        let bb = 2.0042896;
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(bb, sb)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![agent_zero, agent_one];
        let mut rng = StdRng::seed_from_u64(42);

        // Add historian to capture hand history
        let open_hand_hist = Box::new(OpenHandHistoryVecHistorian::new());
        let hand_storage = open_hand_hist.get_storage();
        let vec_hist = Box::new(VecHistorian::new());
        let vec_storage = vec_hist.get_storage();
        let historians: Vec<Box<dyn historian::Historian>> = vec![open_hand_hist, vec_hist];

        let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);

        // Print actions for debugging
        for record in vec_storage.borrow().iter() {
            eprintln!("{:?}", record.action);
        }

        // Check the hand history - there should be no Check actions for all-in players
        // on post-flop streets
        let hands = hand_storage.borrow();
        assert!(!hands.is_empty());
        for hand in hands.iter() {
            assert_valid_open_hand_history(hand);
            assert_open_hand_history_matches_game_state(hand, &sim.game_state);
        }
    }
}
