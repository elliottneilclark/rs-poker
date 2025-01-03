//! This is the core module. It exports the non-holdem
//! related code.

mod error;
pub use self::error::RSPokerError;
/// card.rs has value and suit.
mod card;
/// Re-export Card, Value, and Suit
pub use self::card::{Card, Suit, Value};

/// The bitset hand.
mod hand;
/// Export the hand
pub use self::hand::*;
/// Code related to cards in flattened hands.
mod flat_hand;
/// Everything in there should be public.
pub use self::flat_hand::*;

/// We want to be able to iterate over five card hands.
mod card_iter;
/// Make that functionality public.
pub use self::card_iter::*;

/// Deck is the normal 52 card deck.
mod deck;
/// Export `Deck`
pub use self::deck::Deck;

/// Flattened deck
mod flat_deck;
/// Export the trait and the result.
pub use self::flat_deck::FlatDeck;

/// 5 Card hand ranking code.
mod rank;
/// Export the trait and the results.
pub use self::rank::{Rank, Rankable};

// u16 backed player set.
mod player_bit_set;
// u64 backed card set.
mod card_bit_set;
// Export the bit set and the iterator
pub use self::player_bit_set::{ActivePlayerBitSetIter, PlayerBitSet};
// Export the bit set and the iterator used for cards (52 cards so u64 backed)
pub use self::card_bit_set::{CardBitSet, CardBitSetIter};
