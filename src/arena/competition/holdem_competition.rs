use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
};

use crate::arena::{HoldemSimulation, errors::HoldemSimulationError, game_state::Round};

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

    /// Maximum number of HoldemSimulation's to
    /// keep in a long call to `run`
    max_sim_history: usize,
}

const MAX_PLAYERS: usize = 12;

impl<T: Iterator<Item = HoldemSimulation>> HoldemCompetition<T> {
    /// Creates a new HoldemHandCompetition instance with the provided
    /// HoldemSimulation.
    ///
    /// Initializes the number of rounds to 0 and the stack change vectors to 0
    /// for each agent.
    pub fn new(simulation_iterator: T) -> HoldemCompetition<T> {
        HoldemCompetition {
            simulation_iterator,
            max_sim_history: 100,
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

    pub fn run(
        &mut self,
        num_rounds: usize,
    ) -> Result<Vec<HoldemSimulation>, HoldemSimulationError> {
        let mut sims = VecDeque::with_capacity(self.max_sim_history);
        let mut rand = rand::rng();

        for _round in 0..num_rounds {
            // Createa a new holdem simulation
            let mut running_sim = self.simulation_iterator.next().unwrap();
            // Run the sim
            running_sim.run(&mut rand);
            // Update the stack change stats
            self.update_metrics(&running_sim);
            // Update the counter
            self.num_rounds += 1;
            // If there are too many sims in the circular queue then make some space
            if sims.len() >= self.max_sim_history {
                sims.pop_front();
            }
            // Store the final results
            sims.push_back(running_sim);
        }

        // Drain the whole vecdequeue
        Ok(sims.into_iter().collect())
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
        AgentGenerator, CloneGameStateGenerator,
        agent::{CallingAgentGenerator, RandomAgentGenerator},
        competition::StandardSimulationIterator,
    };

    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    fn test_standard_simulation() {
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

        let _first_results = competition.run(100).unwrap();
    }
}
