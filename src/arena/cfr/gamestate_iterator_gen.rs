use crate::arena::GameState;

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
