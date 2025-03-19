use crate::arena::GameState;

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

/// Creates an iterator that generates `num_hands` clones of the input game state.
///
/// This implementation of [`GameStateIteratorGen`] creates a simple iterator that
/// produces exact copies of the input game state. The number of copies is determined
/// by the `num_hands` field set during construction.
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
        let num_hands = self.num_hands;
        (0..num_hands).map(move |_| game_state.clone())
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
}
