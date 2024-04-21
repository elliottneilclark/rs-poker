use tracing::{event, trace_span};

use super::{errors::HoldemSimulationError, historian::HistorianBuilder, AgentBuilder, GameState};

/// A `SingleTableTournament` is a tournament that has multiple agents
/// playing holdem poker at a single table. The tournament is played
/// until a single agent has all the money.
///
/// This builder is used to create a `SingleTableTournament`.
#[derive(Default)]
pub struct SingleTableTournamentBuilder {
    agent_builders: Option<Vec<Box<dyn AgentBuilder>>>,
    historian_builders: Option<Vec<Box<dyn HistorianBuilder>>>,
    starting_game_state: Option<GameState>,
    panic_on_historian_error: bool,
}

pub struct SingleTableTournament {
    agent_builders: Vec<Box<dyn AgentBuilder>>,
    historian_builders: Vec<Box<dyn HistorianBuilder>>,
    starting_game_state: GameState,
    panic_on_historian_error: bool,
    // TODO should this include payouts?
}

impl SingleTableTournamentBuilder {
    pub fn agent_builders(mut self, agent_builders: Vec<Box<dyn AgentBuilder>>) -> Self {
        self.agent_builders = Some(agent_builders);
        self
    }

    pub fn historian_builders(
        mut self,
        historian_builders: Vec<Box<dyn HistorianBuilder>>,
    ) -> Self {
        self.historian_builders = Some(historian_builders);
        self
    }

    pub fn starting_game_state(mut self, starting_game_state: GameState) -> Self {
        self.starting_game_state = Some(starting_game_state);
        self
    }

    pub fn panic_on_historian_error(mut self, panic_on_historian_error: bool) -> Self {
        self.panic_on_historian_error = panic_on_historian_error;
        self
    }

    pub fn build(self) -> Result<SingleTableTournament, HoldemSimulationError> {
        let agent_builders = self
            .agent_builders
            .ok_or(HoldemSimulationError::NeedAgents)?;
        let starting_game_state = self
            .starting_game_state
            .ok_or(HoldemSimulationError::NeedGameState)?;
        let historian_builders = self.historian_builders.unwrap_or_default();
        Ok(SingleTableTournament {
            agent_builders,
            historian_builders,
            starting_game_state,
            panic_on_historian_error: self.panic_on_historian_error,
        })
    }
}

impl SingleTableTournament {
    // Returns a vector of the place that each agent finished in.
    pub fn run(&self) -> Result<Vec<usize>, HoldemSimulationError> {
        let span = trace_span!("SingleTableTournament::run");
        let _enter = span.enter();

        let mut place = self.agent_builders.len();
        let mut results = vec![0; self.agent_builders.len()];
        let mut game_state = self.starting_game_state.clone();
        while place > 1 {
            let agents = self
                .agent_builders
                .iter()
                .map(|builder| builder.build(&game_state))
                .collect::<Vec<_>>();
            let historians = self
                .historian_builders
                .iter()
                .map(|builder| builder.build(&game_state))
                .collect::<Vec<_>>();
            let mut sim = crate::arena::HoldemSimulationBuilder::default()
                .game_state(game_state.clone())
                .agents(agents)
                .historians(historians)
                .panic_on_historian_error(self.panic_on_historian_error)
                .build()?;

            // Run the simulation
            sim.run();

            // Update the results
            let mut out = sim
                .game_state
                .stacks
                .iter()
                .enumerate()
                .filter(|(_, stack)| **stack == 0.0)
                .filter(|(idx, _)| sim.game_state.starting_stacks[*idx] != 0.0)
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>();

            // Sort by the starting stack going into the hand
            out.sort_by(|a, b| {
                sim.game_state.starting_stacks[*b]
                    .partial_cmp(&sim.game_state.starting_stacks[*a])
                    .unwrap()
                    .reverse()
            });
            for idx in out {
                event!(
                    tracing::Level::INFO,
                    "Agent {} finished in place {}",
                    idx,
                    place
                );
                results[idx] = place;
                place -= 1;
            }
            let mut dealer_idx = (sim.game_state.dealer_idx + 1) % sim.game_state.stacks.len();
            while sim.game_state.stacks[dealer_idx] == 0.0 {
                dealer_idx = (dealer_idx + 1) % sim.game_state.stacks.len();
            }

            game_state = GameState::new(
                sim.game_state.stacks,
                sim.game_state.big_blind,
                sim.game_state.small_blind,
                sim.game_state.ante,
                dealer_idx,
            );
        }

        if place == 1 {
            let winners: Vec<usize> = game_state
                .stacks
                .iter()
                .enumerate()
                .filter(|(_, stack)| **stack > 0.0)
                .map(|(idx, _)| idx)
                .collect();

            if winners.len() != 1 {
                return Err(HoldemSimulationError::NoWinner);
            }

            let idx = winners[0];

            results[idx] = 1;
            println!("Agent {} finished in place 1", idx);
            event!(tracing::Level::INFO, "Agent {} finished in place 1", idx);
        }
        Ok(results)
    }
}
