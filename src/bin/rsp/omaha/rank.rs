use clap::Args;
use rs_poker::core::{CoreRank, RSPokerError, Rankable};
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
    let inner_value = rank.value_bits();

    println!("Hole:  {}", args.hole_cards);
    println!("Board: {}", args.board);
    println!("Rank:  {:?}", core_rank);
    println!("Sub-rank: {}", inner_value);

    Ok(())
}
