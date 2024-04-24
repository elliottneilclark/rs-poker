use tracing::{event, trace_span};

use crate::arena::{
    agent::AgentGenerator, errors::HoldemSimulationError, historian::HistorianGenerator, GameState,
};

/// A `SingleTableTournament` is a tournament that has multiple agents
/// playing holdem poker at a single table. The tournament is played
/// until a single agent has all the money.
///
/// This builder is used to create a `SingleTableTournament`.
#[derive(Default)]
pub struct SingleTableTournamentBuilder {
    agent_generators: Option<Vec<Box<dyn AgentGenerator>>>,
    historian_generators: Option<Vec<Box<dyn HistorianGenerator>>>,
    starting_game_state: Option<GameState>,
    panic_on_historian_error: bool,
}

pub struct SingleTableTournament {
    agent_generators: Vec<Box<dyn AgentGenerator>>,
    historian_generators: Vec<Box<dyn HistorianGenerator>>,
    starting_game_state: GameState,
    panic_on_historian_error: bool,
    // TODO should this include payouts?
}

impl SingleTableTournamentBuilder {
    /// Sets the agent generators for the tournament.
    /// Each generator will be called prior to the start of the game.
    /// The agents will be generated in the order that they are passed in.
    pub fn agent_generators(mut self, agent_generators: Vec<Box<dyn AgentGenerator>>) -> Self {
        self.agent_generators = Some(agent_generators);
        self
    }

    /// Sets the historian generators for the tournament.
    pub fn historian_generators(
        mut self,
        historian_generators: Vec<Box<dyn HistorianGenerator>>,
    ) -> Self {
        self.historian_generators = Some(historian_generators);
        self
    }

    /// Sets the starting game state for the tournament.
    pub fn starting_game_state(mut self, starting_game_state: GameState) -> Self {
        self.starting_game_state = Some(starting_game_state);
        self
    }

    /// Sets whether the underlying `HoldemSimulation` should panic if a
    /// historian errors.
    pub fn panic_on_historian_error(mut self, panic_on_historian_error: bool) -> Self {
        self.panic_on_historian_error = panic_on_historian_error;
        self
    }

    /// Builds the `SingleTableTournament` from the builder.
    pub fn build(self) -> Result<SingleTableTournament, HoldemSimulationError> {
        // Make sure that the needed properties are set
        let agent_builders = self
            .agent_generators
            .ok_or(HoldemSimulationError::NeedAgents)?;
        let starting_game_state = self
            .starting_game_state
            .ok_or(HoldemSimulationError::NeedGameState)?;
        // Historians we default to the empty list
        let historian_builders = self.historian_generators.unwrap_or_default();
        // Return everything
        Ok(SingleTableTournament {
            agent_generators: agent_builders,
            historian_generators: historian_builders,
            starting_game_state,
            panic_on_historian_error: self.panic_on_historian_error,
        })
    }
}

impl SingleTableTournament {
    /// Run the single table tournament to completion.
    ///
    /// Returns a vector of the places that each agent finished in.
    /// From 1 to N where N is the number of agents.
    ///
    /// Meaning `[2 , 1, 3, 4]` indicates that the first agent
    /// finished in second place, the second agent won, the third agent got
    /// third and the fourth agent finished in last.
    pub fn run(self) -> Result<Vec<usize>, HoldemSimulationError> {
        let span = trace_span!("SingleTableTournament::run");
        let _enter = span.enter();

        // The place that we are about to assign to the next agent to bust out.
        let mut place = self.agent_generators.len();
        // Holds the results of the tournament.
        let mut results = vec![0; self.agent_generators.len()];
        let mut game_state = self.starting_game_state;

        // While there is still more than one player left in the tournament
        while place > 1 {
            let agents = self
                .agent_generators
                .iter()
                .map(|builder| builder.generate(&game_state))
                .collect::<Vec<_>>();
            let historians = self
                .historian_generators
                .iter()
                .map(|builder| builder.generate(&game_state))
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

            // For every index that busted out assign the place
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
            // Move the dealer button
            // Find the next player with a stack
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

        // Assign the winner
        if place == 1 {
            let winners: Vec<usize> = game_state
                .stacks
                .iter()
                .enumerate()
                .filter(|(_, stack)| **stack > 0.0)
                .map(|(idx, _)| idx)
                .collect();

            // This should NEVER happen
            if winners.len() != 1 {
                return Err(HoldemSimulationError::NoWinner);
            }

            let idx = winners[0];
            results[idx] = 1;
            event!(tracing::Level::INFO, "Agent {} finished in place 1", idx);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::agent::{AllInAgentGenerator, FoldingAgentGenerator};

    use super::*;

    #[test]
    fn test_all_in() {
        let stacks = vec![50.0; 4];
        let gens: Vec<Box<dyn AgentGenerator>> = vec![
            Box::<AllInAgentGenerator>::default(),
            Box::<AllInAgentGenerator>::default(),
            Box::<AllInAgentGenerator>::default(),
            Box::<AllInAgentGenerator>::default(),
        ];
        let game_state = GameState::new(stacks, 10.0, 5.0, 1.0, 0);
        let tournament = SingleTableTournamentBuilder::default()
            .agent_generators(gens)
            .starting_game_state(game_state)
            .build()
            .unwrap();

        let results = tournament.run().unwrap();

        // Every number 1..4 should be in the results
        for i in 1..4 {
            assert!(results.contains(&i));
        }
    }

    #[test]
    fn test_headsup_tournament_folding_never_wins() {
        let stacks = vec![50.0; 4];

        // The all in agent always raises all in on preflop betting.
        // The Folding Agents will then fold to the bet.
        // Meaning every FoldingAgent loses at least the ante but
        // maybe a small blind or big blind each game.
        //
        // In other words the tournament is gaurnteed to end.
        // meaning that folding agent loses money every round.
        let agent_builders: Vec<Box<dyn AgentGenerator>> = vec![
            Box::<AllInAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
        ];

        let game_state = GameState::new(stacks, 10.0, 5.0, 1.0, 0);

        let tournament = SingleTableTournamentBuilder::default()
            .agent_generators(agent_builders)
            .starting_game_state(game_state)
            .build()
            .unwrap();

        let results = tournament.run().unwrap();

        // Only the calling agent can win.
        assert_eq!(1, results[0]);

        // So everyone else must bust out of the tournament.
        assert!(results[1] > 1);
        assert!(results[2] > 1);
    }
}
