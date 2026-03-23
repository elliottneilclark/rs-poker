use clap::Args;
use rs_poker::core::{CoreRank, FlatHand, Hand, RSPokerError, Rank, RankFive, Rankable};

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

    let rank = if card_count == 5 {
        let flat = FlatHand::new_from_str(&args.cards)?;
        flat.rank_five()
    } else {
        let flat = FlatHand::new_from_str(&args.cards)?;
        flat.rank()
    };

    let core_rank: CoreRank = rank.into();
    let inner_value = match rank {
        Rank::HighCard(v)
        | Rank::OnePair(v)
        | Rank::TwoPair(v)
        | Rank::ThreeOfAKind(v)
        | Rank::Straight(v)
        | Rank::Flush(v)
        | Rank::FullHouse(v)
        | Rank::FourOfAKind(v)
        | Rank::StraightFlush(v) => v,
    };

    println!("Hand:  {}", args.cards);
    println!("Rank:  {:?}", core_rank);
    println!("Value: {}", inner_value);

    Ok(())
}
