use rand::{thread_rng, Rng};

use crate::arena::{historian::Historian, Agent, GameState};

pub trait GameStateGenerator {
    fn generate(&mut self) -> GameState;
}

/// This is a simple generator that just clones the game state
/// every time it's called.
///
/// This holds the dealt cards constant and the stack sizes constant.
pub struct CloneGameStateGenerator {
    game_state: GameState,
}

impl CloneGameStateGenerator {
    pub fn new(game_state: GameState) -> CloneGameStateGenerator {
        CloneGameStateGenerator { game_state }
    }
}

impl GameStateGenerator for CloneGameStateGenerator {
    fn generate(&mut self) -> GameState {
        self.game_state.clone()
    }
}

/// This `GameStateGenerator` generates a random game state with no cards dealt
/// and random stack sizes. The dealer button is also randomly placed.
pub struct RandomGameStateGenerator {
    num_players: usize,
    min_stack: f32,
    max_stack: f32,
    big_blind: f32,
    small_blind: f32,
    ante: f32,
}

impl RandomGameStateGenerator {
    pub fn new(
        num_players: usize,
        min_stack: f32,
        max_stack: f32,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
    ) -> RandomGameStateGenerator {
        RandomGameStateGenerator {
            num_players,
            min_stack,
            max_stack,
            big_blind,
            small_blind,
            ante,
        }
    }
}

impl GameStateGenerator for RandomGameStateGenerator {
    fn generate(&mut self) -> GameState {
        let mut rng = thread_rng();
        let stacks: Vec<f32> = (0..self.num_players)
            .map(|_| rng.gen_range(self.min_stack..self.max_stack))
            .collect();

        let num_players = stacks.len();

        GameState::new(
            stacks,
            self.big_blind,
            self.small_blind,
            self.ante,
            rng.gen_range(0..num_players),
        )
    }
}

pub trait AgentsGenerator {
    fn generate(&mut self, game_state: &GameState) -> Vec<Box<dyn Agent>>;
}

/// This is a trait to make it possible to clone
/// agents that are passed in as `Box<dyn Agent>`
pub trait CloneAgent: Agent {
    fn clone_box(&self) -> Box<dyn Agent>;
}

impl<T> CloneAgent for T
where
    T: 'static + Agent + Clone,
{
    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }
}

/// Generate agents by cloning the ones provided
pub struct CloningAgentsGenerator {
    agents: Vec<Box<dyn CloneAgent>>,
}

impl CloningAgentsGenerator {
    pub fn new(agents: Vec<Box<dyn CloneAgent>>) -> CloningAgentsGenerator {
        CloningAgentsGenerator { agents }
    }
}

impl AgentsGenerator for CloningAgentsGenerator {
    /// Generate agents by cloning the ones provided
    ///
    /// # Example
    /// ```
    /// use rs_poker::arena::agent::FoldingAgent;
    /// use rs_poker::arena::competition::AgentsGenerator;
    /// use rs_poker::arena::competition::CloneAgent;
    /// use rs_poker::arena::competition::CloningAgentsGenerator;
    /// use rs_poker::arena::game_state::GameState;
    ///
    /// let agents: Vec<Box<dyn CloneAgent>> = vec![
    ///     Box::<FoldingAgent>::default(),
    ///     Box::<FoldingAgent>::default(),
    /// ];
    ///
    /// let stacks = vec![100.0, 100.0];
    /// let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
    /// let mut sim_gen = CloningAgentsGenerator::new(agents);
    ///
    /// let agents = sim_gen.generate(&game_state);
    /// assert_eq!(agents.len(), 2);
    ///
    /// let re_agents = sim_gen.generate(&game_state);
    /// assert_eq!(re_agents.len(), 2);
    /// ```
    fn generate(&mut self, _game_state: &GameState) -> Vec<Box<dyn Agent>> {
        self.agents.iter().map(|a| a.clone_box()).collect()
    }
}

pub trait HistorianGenerator {
    fn generate(
        &mut self,
        game_state: &GameState,
        agents: &[Box<dyn Agent>],
    ) -> Vec<Box<dyn Historian>>;
}

pub trait CloneHistorian: Historian {
    fn clone_box(&self) -> Box<dyn Historian>;
}

impl<T> CloneHistorian for T
where
    T: 'static + Historian + Clone,
{
    fn clone_box(&self) -> Box<dyn Historian> {
        Box::new(self.clone())
    }
}

pub struct CloningHistorianGenerator {
    historians: Vec<Box<dyn CloneHistorian>>,
}

impl CloningHistorianGenerator {
    pub fn new(historians: Vec<Box<dyn CloneHistorian>>) -> CloningHistorianGenerator {
        CloningHistorianGenerator { historians }
    }
}

impl HistorianGenerator for CloningHistorianGenerator {
    fn generate(
        &mut self,
        _game_state: &GameState,
        _agents: &[Box<dyn Agent>],
    ) -> Vec<Box<dyn Historian>> {
        self.historians.iter().map(|h| h.clone_box()).collect()
    }
}

/// This is a simple generator that just returns an empty vector of historians
///
/// This is useful for simulations where no historians are needed.
pub struct EmptyHistorianGenerator;

impl HistorianGenerator for EmptyHistorianGenerator {
    fn generate(
        &mut self,
        _game_state: &GameState,
        _agents: &[Box<dyn Agent>],
    ) -> Vec<Box<dyn Historian>> {
        vec![]
    }
}
