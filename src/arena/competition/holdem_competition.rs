use std::{collections::VecDeque, fmt::Debug};

use crate::arena::{errors::HoldemSimulationError, HoldemSimulation};

use super::sim_gen::HoldemSimulationGenerator;

/// A  struct to help seeing which agent is likely to do well
pub struct HoldemCompetition<T: HoldemSimulationGenerator> {
    sim_gen: T,
    /// The number of rounds that have been run.
    pub num_rounds: usize,

    /// stack size change normalized in big blinds
    pub total_change: Vec<f32>,
    pub max_change: Vec<f32>,
    pub min_change: Vec<f32>,

    /// How many times each agent sees showdown
    pub showdown_count: Vec<usize>,
    /// How many hands each agent has made some profit
    pub win_count: Vec<usize>,
    /// How many hands the agents have lost money
    pub loss_count: Vec<usize>,

    /// Maximum number of HoldemSimulation's to
    /// keep in a long call to `run`
    pub max_sim_history: usize,
}

impl<T: HoldemSimulationGenerator> HoldemCompetition<T>
where
    T: Clone,
{
    /// Creates a new HoldemHandCompetition instance with the provided
    /// HoldemSimulation.
    ///
    /// Initializes the number of rounds to 0 and the stack change vectors to 0
    /// for each agent.
    pub fn new(gen: T) -> HoldemCompetition<T> {
        let num_agents = gen.num_agents();
        HoldemCompetition {
            sim_gen: gen,
            max_sim_history: 100,
            // Set everything to zero
            num_rounds: 0,
            total_change: vec![0.0; num_agents],
            min_change: vec![0.0; num_agents],
            max_change: vec![0.0; num_agents],
            showdown_count: vec![0; num_agents],
            win_count: vec![0; num_agents],
            loss_count: vec![0; num_agents],
        }
    }

    pub fn run(
        &mut self,
        num_rounds: usize,
    ) -> Result<Vec<HoldemSimulation>, HoldemSimulationError> {
        let mut sims = VecDeque::with_capacity(self.max_sim_history);

        for (_round, mut running_sim) in (0..num_rounds).zip(self.sim_gen.clone()) {
            // Now that we have our own memory run the whole simulation to completion
            running_sim.run();
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
            if norm_change.is_sign_positive() {
                self.win_count[idx] += 1;
            } else if norm_change.is_sign_negative() {
                self.loss_count[idx] += 1;
            }
        }
    }
}
impl<T: HoldemSimulationGenerator> Debug for HoldemCompetition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldemCompetition")
            .field("num_rounds", &self.num_rounds)
            .field("total_change", &self.total_change)
            .field("max_change", &self.max_change)
            .field("min_change", &self.min_change)
            .field("showdown_count", &self.showdown_count)
            .field("win_count", &self.win_count)
            .field("loss_count", &self.loss_count)
            .field("max_sim_history", &self.max_sim_history)
            .finish()
    }
}
