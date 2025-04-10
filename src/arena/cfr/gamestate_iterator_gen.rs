use crate::arena::{GameState, game_state};

/// This trait defines an interface for types that can generate an iterator
/// over possible game states from a given initial game state.
pub trait GameStateIteratorGen {
    fn generate(&self, game_state: &GameState) -> impl Iterator<Item = GameState>;
}

#[derive(Clone, Debug)]
pub struct FixedGameStateIteratorGen {
    pub num_hands: usize,
}

impl FixedGameStateIteratorGen {
    pub fn new(num_hands: usize) -> Self {
        Self { num_hands }
    }
}

impl Default for FixedGameStateIteratorGen {
    fn default() -> Self {
        Self { num_hands: 10 }
    }
}

/// Creates an iterator that generates `num_hands` clones of the input game
/// state.
///
/// This implementation of [`GameStateIteratorGen`] creates a simple iterator
/// that produces exact copies of the input game state. The number of copies is
/// determined by the `num_hands` field set during construction.
///
/// # Arguments
///
/// * `game_state` - The game state to be cloned
///
/// # Returns
///
/// Returns an iterator that yields `num_hands` clones of the input `game_state`
impl GameStateIteratorGen for FixedGameStateIteratorGen {
    fn generate(&self, game_state: &GameState) -> impl Iterator<Item = GameState> {
        (0..self.num_hands).map(|_| game_state.clone())
    }
}

#[derive(Clone, Debug)]
pub struct PerRoundFixedGameStateIteratorGen {
    // For PreFlop
    pub pre_flop_num_hands: usize,
    // For Flop
    pub flop_num_hands: usize,
    // For Turn
    pub turn_num_hands: usize,
    // For River
    pub river_num_hands: usize,
}

impl PerRoundFixedGameStateIteratorGen {
    pub fn new(
        pre_flop_num_hands: usize,
        flop_num_hands: usize,
        turn_num_hands: usize,
        river_num_hands: usize,
    ) -> Self {
        Self {
            pre_flop_num_hands,
            flop_num_hands,
            turn_num_hands,
            river_num_hands,
        }
    }

    fn num_hands(&self, game_state: &GameState) -> usize {
        match game_state.round {
            game_state::Round::Preflop => self.pre_flop_num_hands,
            game_state::Round::Flop => self.flop_num_hands,
            game_state::Round::Turn => self.turn_num_hands,
            game_state::Round::River => self.river_num_hands,
            _ => 1, // Handle any other rounds if necessary
        }
    }
}

impl Default for PerRoundFixedGameStateIteratorGen {
    fn default() -> Self {
        Self {
            pre_flop_num_hands: 10,
            flop_num_hands: 10,
            turn_num_hands: 10,
            river_num_hands: 1,
        }
    }
}

impl GameStateIteratorGen for PerRoundFixedGameStateIteratorGen {
    fn generate(&self, game_state: &GameState) -> impl Iterator<Item = GameState> {
        (0..self.num_hands(game_state)).map(|_| game_state.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() {
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let generator = FixedGameStateIteratorGen::new(3);
        let mut iter = generator.generate(&game_state);

        assert_eq!(iter.next().unwrap(), game_state);
        assert_eq!(iter.next().unwrap(), game_state);
        assert_eq!(iter.next().unwrap(), game_state);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_per_round() {
        let mut game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let generator = PerRoundFixedGameStateIteratorGen::new(2, 3, 4, 1);

        game_state.advance_round();
        game_state.advance_round();
        game_state.advance_round();

        assert_eq!(game_state.round, game_state::Round::Preflop);

        // Preflop
        {
            let mut iter = generator.generate(&game_state);
            assert_eq!(iter.next().unwrap(), game_state);
            assert_eq!(iter.next().unwrap(), game_state);
            assert!(iter.next().is_none());
        }

        // Flop
        game_state.advance_round();
        game_state.advance_round();
        assert_eq!(game_state.round, game_state::Round::Flop);
        {
            let mut iter_flop = generator.generate(&game_state);
            assert_eq!(iter_flop.next().unwrap(), game_state);
            assert_eq!(iter_flop.next().unwrap(), game_state);
            assert_eq!(iter_flop.next().unwrap(), game_state);
            assert!(iter_flop.next().is_none());
        }

        // Turn
        game_state.advance_round();
        game_state.advance_round();
        assert_eq!(game_state.round, game_state::Round::Turn);
        {
            let mut iter_turn = generator.generate(&game_state);

            assert_eq!(iter_turn.next().unwrap(), game_state);
            assert_eq!(iter_turn.next().unwrap(), game_state);
            assert_eq!(iter_turn.next().unwrap(), game_state);
            assert_eq!(iter_turn.next().unwrap(), game_state);
            assert!(iter_turn.next().is_none());
        }

        // River
        game_state.advance_round();
        game_state.advance_round();
        assert_eq!(game_state.round, game_state::Round::River);
        {
            let mut iter_river = generator.generate(&game_state);

            assert_eq!(iter_river.next().unwrap(), game_state);
            assert!(iter_river.next().is_none());
        }
    }
}
