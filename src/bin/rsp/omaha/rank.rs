use clap::Args;
use rs_poker::core::{CoreRank, RSPokerError, Rank, Rankable};
use rs_poker::omaha::OmahaHand;

#[derive(Debug, thiserror::Error)]
pub enum OmahaRankError {
    #[error(transparent)]
    Poker(#[from] RSPokerError),
}

#[derive(Args)]
pub struct RankArgs {
    /// Hole cards (4-7 cards, e.g. "AhAsKhKs")
    hole_cards: String,

    /// Board cards (3-5 cards, e.g. "QhJhTh9h8h")
    board: String,
}

pub fn run(args: RankArgs) -> Result<(), OmahaRankError> {
    let hand = OmahaHand::new_from_str(&args.hole_cards, &args.board)?;
    let rank = hand.rank();

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

    println!("Hole:  {}", args.hole_cards);
    println!("Board: {}", args.board);
    println!("Rank:  {:?}", core_rank);
    println!("Value: {}", inner_value);

    Ok(())
}
