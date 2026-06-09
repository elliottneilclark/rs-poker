use clap::Args;
use rs_poker::core::{CoreRank, FlatHand, Hand, RSPokerError, Rankable};

#[derive(Debug, thiserror::Error)]
pub enum RankError {
    #[error(transparent)]
    Poker(#[from] RSPokerError),
    #[error("Expected 5-7 cards, got {0}. Example: AcKdQhJsTs")]
    InvalidCardCount(usize),
}

#[derive(Args)]
pub struct RankArgs {
    /// Cards to evaluate (5-7 cards, e.g. "AcKdQhJsTs")
    cards: String,
}

pub fn run(args: RankArgs) -> Result<(), RankError> {
    let hand = Hand::new_from_str(&args.cards)?;
    let card_count = hand.count();

    if !(5..=7).contains(&card_count) {
        return Err(RankError::InvalidCardCount(card_count));
    }

    let rank = FlatHand::new_from_str(&args.cards)?.rank();

    let core_rank: CoreRank = rank.into();
    let inner_value = rank.value_bits();

    println!("Hand:  {}", args.cards);
    println!("Rank:  {:?}", core_rank);
    println!("Sub-rank: {}", inner_value);

    Ok(())
}
