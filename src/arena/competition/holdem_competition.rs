use std::{collections::HashMap, fmt::Debug, sync::Arc};

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::arena::{
    HoldemSimulation,
    errors::HoldemSimulationError,
    game_state::{MAX_PLAYERS, Round},
};

/// A  struct to help seeing which agent is likely to do well
///
/// Each competition is a series of `HoldemSimulations`
/// from the `HoldemSimulationGenerator` passed in.
pub struct HoldemCompetition<T: Iterator<Item = HoldemSimulation>> {
    simulation_iterator: T,
    /// The number of rounds that have been run.
    pub num_rounds: usize,

    /// stack size change normalized in big blinds
    pub total_change: Vec<f32>,
    pub max_change: Vec<f32>,
    pub min_change: Vec<f32>,

    /// How many hands each agent has made some profit
    pub win_count: Vec<usize>,
    /// How many hands the agents have lost money
    pub loss_count: Vec<usize>,
    // How many times the agent has lost no money
    pub zero_count: Vec<usize>,
    // Count of the round before the simulation stopped
    pub before_count: HashMap<Round, usize>,

    /// In-flight cap on the number of *simulations* this competition runs
    /// concurrently. This is the competition's own limiter; it is independent
    /// of the limiter each CFR agent uses for its exploration unless a caller
    /// deliberately threads the same `Arc<Semaphore>` into both (via
    /// `CFRAgentBuilder::limiter`). It is not shared automatically.
    limiter: Arc<Semaphore>,
}

impl<T: Iterator<Item = HoldemSimulation>> HoldemCompetition<T> {
    /// Creates a new HoldemHandCompetition instance with the provided
    /// HoldemSimulation.
    ///
    /// Initializes the number of rounds to 0 and the stack change vectors to 0
    /// for each agent. Each simulation owns and deals from its own RNG (seeded
    /// by the iterator), so the competition holds no RNG of its own.
    pub fn new(simulation_iterator: T) -> HoldemCompetition<T> {
        HoldemCompetition {
            simulation_iterator,
            limiter: crate::arena::cfr::build_default_limiter(),
            // Set everything to zero
            num_rounds: 0,
            total_change: vec![0.0; MAX_PLAYERS],
            min_change: vec![0.0; MAX_PLAYERS],
            max_change: vec![0.0; MAX_PLAYERS],
            win_count: vec![0; MAX_PLAYERS],
            loss_count: vec![0; MAX_PLAYERS],
            zero_count: vec![0; MAX_PLAYERS],
            // Round before stopping
            before_count: HashMap::new(),
        }
    }

    /// Run `num_rounds` simulations and return every completed one.
    ///
    /// Simulations are spawned onto the runtime and run concurrently, with
    /// `limiter` capping how many are *in flight* at once — this bounds peak
    /// CPU/inference pressure (and lets a caller share that pressure budget
    /// with the agents' own exploration) without serializing the rounds.
    /// Finished simulations are drained each iteration so the `JoinSet` does
    /// not accumulate completed results, and their per-player metrics are
    /// folded into the competition as they land.
    ///
    /// The returned `Vec` contains all `num_rounds` simulations, each retaining
    /// its full final state and history so callers can inspect every hand.
    /// Memory therefore scales linearly with `num_rounds`; size the call
    /// accordingly when running long competitions.
    ///
    /// # Errors
    ///
    /// Returns a [`HoldemSimulationError`] if a spawned simulation fails to
    /// join; a panic inside a simulation is re-raised on the calling thread.
    pub async fn run(
        &mut self,
        num_rounds: usize,
    ) -> Result<Vec<HoldemSimulation>, HoldemSimulationError> {
        let mut set: JoinSet<HoldemSimulation> = JoinSet::new();
        let mut completed = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            let mut sim = self.simulation_iterator.next().unwrap();
            // Permit acquisition bounds the number of *running* sims; draining
            // finished ones each iteration keeps the JoinSet from holding
            // completed results until the very end.
            let permit = self.limiter.clone().acquire_owned().await.unwrap();
            set.spawn(async move {
                let _permit = permit; // held for the sim's lifetime
                sim.run().await;
                sim
            });
            while let Some(joined) = set.try_join_next() {
                self.record(joined, &mut completed)?;
            }
        }

        while let Some(joined) = set.join_next().await {
            self.record(joined, &mut completed)?;
        }

        Ok(completed)
    }

    /// Fold one completed simulation into the running metrics and the result
    /// vector, surfacing a join failure as an error.
    fn record(
        &mut self,
        joined: Result<HoldemSimulation, tokio::task::JoinError>,
        completed: &mut Vec<HoldemSimulation>,
    ) -> Result<(), HoldemSimulationError> {
        let sim = match joined {
            Ok(sim) => sim,
            // A `JoinError` from a panicking spawned simulation is a bug to
            // surface, not swallow: re-raise the original panic on this thread
            // (mirrors the CFR exploration engine). Non-panic join errors
            // (e.g. an aborted task) still convert to a recoverable error.
            Err(join_err) => {
                if join_err.is_panic() {
                    std::panic::resume_unwind(join_err.into_panic());
                }
                return Err(HoldemSimulationError::from(join_err));
            }
        };
        self.update_metrics(&sim);
        self.num_rounds += 1;
        completed.push(sim);
        Ok(())
    }

    fn update_metrics(&mut self, running_sim: &HoldemSimulation) {
        // Calculates the change in each player's winnings for the round,
        // normalized by the big blind amount.
        //
        // TODO: we need to filter out the players that never started the hand.
        let changes = running_sim
            .game_state
            .starting_stacks
            .iter()
            .zip(running_sim.game_state.stacks.iter())
            .enumerate()
            .map(|(idx, (starting, ending))| {
                (
                    idx,
                    (*ending - *starting) / running_sim.game_state.big_blind,
                )
            });

        for (idx, norm_change) in changes {
            // Running total
            self.total_change[idx] += norm_change;
            // What's the most we lose
            self.min_change[idx] = self.min_change[idx].min(norm_change);
            // What's the most we win
            self.max_change[idx] = self.max_change[idx].max(norm_change);

            // Count how many times the agent wins or loses
            if norm_change > 0.0 {
                self.win_count[idx] += 1;
            } else if norm_change < 0.0 {
                self.loss_count[idx] += 1;
            } else {
                self.zero_count[idx] += 1;
            }
        }
        // Update the count
        let count = self
            .before_count
            .entry(running_sim.game_state.round_before)
            .or_default();
        *count += 1;
    }
}

impl<T: Iterator<Item = HoldemSimulation>> Debug for HoldemCompetition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldemCompetition")
            .field("num_rounds", &self.num_rounds)
            .field("total_change", &self.total_change)
            .field("max_change", &self.max_change)
            .field("min_change", &self.min_change)
            .field("win_count", &self.win_count)
            .field("zero_count", &self.zero_count)
            .field("loss_count", &self.loss_count)
            .field("round_before", &self.before_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        Agent, AgentGenerator, CloneGameStateGenerator,
        agent::{CallingAgentGenerator, RandomAgentGenerator},
        competition::StandardSimulationIterator,
    };

    use super::*;
    use crate::arena::GameStateBuilder;

    #[tokio::test]
    async fn test_standard_simulation() {
        let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
            Box::<RandomAgentGenerator>::default(),
            Box::<CallingAgentGenerator>::default(),
        ];

        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let sim_gen = StandardSimulationIterator::new(
            agent_gens,
            vec![], // no historians
            CloneGameStateGenerator::new(game_state),
        );
        let mut competition = HoldemCompetition::new(sim_gen);

        let _first_results = competition.run(100).await.unwrap();
    }

    #[tokio::test]
    async fn test_thirteen_player_competition() {
        // Regression: HoldemCompetition used to hardcode MAX_PLAYERS = 12,
        // so a 13+ player simulation would index out of bounds when
        // updating per-player metrics. Use 13 (one past the old cap) so
        // the bug triggers without requiring a 16-player game.
        const NUM_PLAYERS: usize = 13;
        let agent_gens: Vec<Box<dyn AgentGenerator>> = (0..NUM_PLAYERS)
            .map(|_| Box::<CallingAgentGenerator>::default() as Box<dyn AgentGenerator>)
            .collect();

        let game_state = GameStateBuilder::new()
            .num_players_with_stack(NUM_PLAYERS, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let sim_gen = StandardSimulationIterator::new(
            agent_gens,
            vec![],
            CloneGameStateGenerator::new(game_state),
        );
        let mut competition = HoldemCompetition::new(sim_gen);

        let _results = competition.run(5).await.unwrap();
        assert_eq!(MAX_PLAYERS, competition.total_change.len());
    }

    #[tokio::test]
    #[should_panic(expected = "boom: agent panicked")]
    async fn test_spawned_sim_panic_is_resurfaced() {
        // Regression: a panic inside a spawned simulation must be re-raised by
        // the competition (resume_unwind), not silently converted to a
        // recoverable `HoldemSimulationError` and swallowed.
        use crate::arena::GameState;
        use crate::arena::action::AgentAction;
        use async_trait::async_trait;

        struct PanicAgent;

        #[async_trait]
        impl Agent for PanicAgent {
            async fn act(&mut self, _id: u128, _game_state: &GameState) -> AgentAction {
                panic!("boom: agent panicked");
            }
            fn name(&self) -> &str {
                "PanicAgent"
            }
        }

        struct PanicAgentGenerator;
        impl AgentGenerator for PanicAgentGenerator {
            fn generate(&self, _player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
                Box::new(PanicAgent)
            }
        }

        let agent_gens: Vec<Box<dyn AgentGenerator>> =
            vec![Box::new(PanicAgentGenerator), Box::new(PanicAgentGenerator)];
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let sim_gen = StandardSimulationIterator::new(
            agent_gens,
            vec![],
            CloneGameStateGenerator::new(game_state),
        );
        let mut competition = HoldemCompetition::new(sim_gen);

        let _ = competition.run(1).await;
    }
}
