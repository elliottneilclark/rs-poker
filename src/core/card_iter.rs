use core::*;

/// Given some cards create sets of possible groups of cards.
#[derive(Debug)]
pub struct CardIter<'a> {
    /// All the possible cards that can be dealt
    possible_cards: &'a [Card],

    /// Set of current offsets being used to create card sets.
    idx: Vec<usize>,

    /// size of card sets requested.
    num_cards: usize,
}


/// `CardIter` is a container for cards and current state.
impl<'a> CardIter<'a> {
    /// Create a new `CardIter` from a slice of cards.
    /// `num_cards` represents how many cards should be in the resulting vector.
    pub fn new(possible_cards: &[Card], num_cards: usize) -> CardIter {
        let mut idx: Vec<usize> = (0..(num_cards as usize)).collect();
        if num_cards > 1 {
            idx[num_cards - 1] -= 1;
        }
        CardIter {
            possible_cards: possible_cards,
            idx: idx,
            num_cards: num_cards,
        }
    }
}

/// The actual `Iterator` for `Card`'s.
impl<'a> Iterator for CardIter<'a> {
    type Item = Vec<Card>;
    fn next(&mut self) -> Option<Vec<Card>> {
        // This is a complete hack.
        //
        // Basically if num_cards == 1 then CardIter::new couldn't
        // set the last index to one less than the starting index,
        // because doing so would cause the unsigend usize to roll over.
        // That means that we need this hack here.
        if self.num_cards == 1 {
            if self.idx[0] < self.possible_cards.len() {
                let c = self.possible_cards[self.idx[0]];
                self.idx[0] += 1;
                return Some(vec![c]);
            } else {
                return None;
            }
        }
        // Keep track of where we are mutating
        let mut current_level: usize = self.num_cards - 1;

        while current_level < self.num_cards {
            // Move the current level forward one.
            self.idx[current_level] += 1;

            // Now check if moving this level forward means that
            // We will need more cards to fill out the rest of the hand
            // then are there.
            let cards_needed_after = self.num_cards - (current_level + 1);
            if self.idx[current_level] + cards_needed_after >= self.possible_cards.len() {
                if current_level == 0 {
                    return None;
                }
                current_level -= 1;
            } else {
                // If we aren't at the end then
                if current_level < self.num_cards - 1 {
                    self.idx[current_level + 1] = self.idx[current_level];
                }
                // Move forward one level
                current_level += 1;
            }
        }

        let result_cards: Vec<Card> = self.idx
            .iter()
            .map(|i| self.possible_cards[*i as usize])
            .collect();
        Some(result_cards)
    }
}

/// This is useful for trying every possible 5 card hand
///
/// Probably not something that's going to be done in real
/// use cases, but still not bad.
impl<'a> IntoIterator for &'a FlatDeck {
    type Item = Vec<Card>;
    type IntoIter = CardIter<'a>;

    fn into_iter(self) -> CardIter<'a> {
        CardIter::new(&self[..], 5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::card::Card;
    use core::hand::Hand;
    use core::deck::*;
    use core::flat_deck::*;

    #[test]
    fn test_iter_one() {
        let mut h = Hand::default();
        h.push(Card {
            value: Value::Two,
            suit: Suit::Spade,
        });

        for cards in CardIter::new(&h[..], 1) {
            assert_eq!(1, cards.len());
        }
        assert_eq!(1, CardIter::new(&h[..], 1).count());
    }

    #[test]
    fn test_iter_two() {
        let mut h = Hand::default();
        h.push(Card {
            value: Value::Two,
            suit: Suit::Spade,
        });
        h.push(Card {
            value: Value::Three,
            suit: Suit::Spade,
        });
        h.push(Card {
            value: Value::Four,
            suit: Suit::Spade,
        });

        // Make sure that we get the correct number back.
        assert_eq!(3, CardIter::new(&h[..], 2).count());

        // Make sure that everything has two cards and they are different.
        //
        for cards in CardIter::new(&h[..], 2) {
            assert_eq!(2, cards.len());
            assert!(cards[0] != cards[1]);
        }
    }

    #[test]
    fn test_iter_deck() {
        let d: FlatDeck = Deck::default().into();
        assert_eq!(2598960, d.into_iter().count());
    }
}
