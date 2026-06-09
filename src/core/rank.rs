//! Hand ranking: turning cards into a comparable [`Rank`].
//!
//! A [`Rank`] is the score produced by the perfect-hash evaluator whose lookup
//! tables are generated in `build.rs` (read that file's header for the full
//! algorithm). This module is the public face of that evaluator: the [`Rank`]
//! and [`CoreRank`] types that wrap a score, the [`Rankable`] trait that ranks a
//! hand, and [`SevenCardAccum`] for ranking many hands that share cards.
//!
//! # The score
//!
//! A `Rank` is a single `u16` laid out as `(category << 12) | subrank`:
//!
//! ```text
//!   bit:  15 .. 12 | 11 ........ 0
//!         category  | subrank
//!         (1..=9)   | (0..=4095)
//! ```
//!
//! `category` is the hand class (1 = high card up to 9 = straight flush) and
//! `subrank` breaks ties within a class. Because the category sits in the high
//! bits, a plain `u16` comparison yields the correct poker ordering, and every
//! distinct five-card hand collapses onto one of exactly 7462 scores. The
//! `build.rs` header explains where that 7462 comes from and how a hand is
//! mapped onto its score.
//!
//! # Ranking a hand
//!
//! You implement nothing yourself: [`Rankable`] is implemented for the common
//! card containers ([`FlatHand`], [`Hand`], [`CardBitSet`], `Vec<Card>`,
//! `[Card]`). Call [`rank`](Rankable::rank) to get the best five-card `Rank`,
//! whether the container holds five, seven, or anything in between. For hot
//! loops that rank many hands sharing a common prefix, reach for
//! [`SevenCardAccum`].

use crate::core::card::Card;

use super::eval;
use super::{CardBitSet, FlatHand, Hand};

/// A packed, comparable poker hand rank.
///
/// Wraps the evaluator's `u16` score `(category << 12) | subrank` (see the
/// module docs). Higher is better, and because the category occupies the high
/// bits, ordering is a single `u16` comparison. Serializes transparently as that
/// bare `u16`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
#[repr(transparent)]
pub struct Rank(u16);

/// Category band shift: `value >> CATEGORY_SHIFT` is the category.
const CATEGORY_SHIFT: u32 = 12;

impl Rank {
    /// Construct from a raw evaluator score. Crate-internal; the evaluator is
    /// the only producer.
    #[inline]
    pub(crate) const fn from_score(score: u16) -> Self {
        Rank(score)
    }
    /// Lowest possible value in each category band, for range comparisons.
    pub const HIGH_CARD_MIN: Rank = Rank(1 << CATEGORY_SHIFT);
    pub const ONE_PAIR_MIN: Rank = Rank(2 << CATEGORY_SHIFT);
    pub const TWO_PAIR_MIN: Rank = Rank(3 << CATEGORY_SHIFT);
    pub const THREE_OF_A_KIND_MIN: Rank = Rank(4 << CATEGORY_SHIFT);
    pub const STRAIGHT_MIN: Rank = Rank(5 << CATEGORY_SHIFT);
    pub const FLUSH_MIN: Rank = Rank(6 << CATEGORY_SHIFT);
    pub const FULL_HOUSE_MIN: Rank = Rank(7 << CATEGORY_SHIFT);
    pub const FOUR_OF_A_KIND_MIN: Rank = Rank(8 << CATEGORY_SHIFT);
    pub const STRAIGHT_FLUSH_MIN: Rank = Rank(9 << CATEGORY_SHIFT);

    /// The hand category, kicker detail stripped.
    #[inline]
    pub const fn category(self) -> CoreRank {
        match self.0 >> CATEGORY_SHIFT {
            1 => CoreRank::HighCard,
            2 => CoreRank::OnePair,
            3 => CoreRank::TwoPair,
            4 => CoreRank::ThreeOfAKind,
            5 => CoreRank::Straight,
            6 => CoreRank::Flush,
            7 => CoreRank::FullHouse,
            8 => CoreRank::FourOfAKind,
            // 9 is StraightFlush; values outside 1..=9 only arise from a
            // malformed deserialize and fall here.
            _ => CoreRank::StraightFlush,
        }
    }
    /// The dense sub-rank within the category (low 12 bits).
    #[inline]
    pub const fn value_bits(self) -> u16 {
        self.0 & ((1 << CATEGORY_SHIFT) - 1)
    }
}

impl std::fmt::Debug for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}({})", self.category(), self.value_bits())
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum CoreRank {
    HighCard,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
}

impl std::fmt::Display for CoreRank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighCard => write!(f, "High Card"),
            Self::OnePair => write!(f, "One Pair"),
            Self::TwoPair => write!(f, "Two Pair"),
            Self::ThreeOfAKind => write!(f, "Three of a Kind"),
            Self::Straight => write!(f, "Straight"),
            Self::Flush => write!(f, "Flush"),
            Self::FullHouse => write!(f, "Full House"),
            Self::FourOfAKind => write!(f, "Four of a Kind"),
            Self::StraightFlush => write!(f, "Straight Flush"),
        }
    }
}

/// Convert from Rank to CoreRank by stripping the u32 detail.
/// This is useful to reduce the cardinality of ranks.
///
/// For example displaying the possible outcomes of a hand
/// without caring about the specific rank values.
impl From<Rank> for CoreRank {
    fn from(rank: Rank) -> Self {
        rank.category()
    }
}

/// Rank a hand, returning the best five-card [`Rank`] it makes.
///
/// This is the single entry point for ranking. It is implemented for the common
/// card containers ([`FlatHand`], [`Hand`], [`CardBitSet`], `Vec<Card>`,
/// `[Card]`), which fold their cards through the perfect-hash evaluator, and for
/// game-specific hands such as [`OmahaHand`](crate::omaha::OmahaHand), which
/// enumerate their legal five-card combinations and take the best.
///
/// It works for any card count the evaluator supports (up to seven); five-card
/// containers are simply the common case, ranked in a single lookup.
///
/// # Examples
/// ```
/// use rs_poker::core::{CoreRank, FlatHand, Rankable};
///
/// // Best five of seven: two pair (eights and twos) with a king kicker.
/// let hand = FlatHand::new_from_str("2h2d8d8sKd6sTh").unwrap();
/// assert_eq!(hand.rank().category(), CoreRank::TwoPair);
///
/// // A five-card flush.
/// let flush = FlatHand::new_from_str("Ad8d9dTd5d").unwrap();
/// assert_eq!(flush.rank().category(), CoreRank::Flush);
/// ```
pub trait Rankable {
    fn rank(&self) -> Rank;
}

/// Incremental "best 5 of N" hand ranker for hot loops over hands that share
/// cards.
///
/// # Why use it
///
/// Ranking a 7-card hand all at once (e.g. [`Rankable::rank`]) re-folds every
/// card on every call. When you rank many hands that share a common prefix,
/// such as board enumeration or equity rollouts, that repeated folding
/// dominates. `SevenCardAccum` lets you tally the shared cards **once** and
/// extend cheaply.
///
/// # How to use it
///
/// Create one, [`add`](Self::add) each card, then call [`rank`](Self::rank) for
/// the best 5-card [`Rank`]. The result equals ranking all the added cards
/// together. Add at least five cards before calling `rank`.
///
/// ```
/// use rs_poker::core::{Card, SevenCardAccum, Suit, Value};
///
/// let mut acc = SevenCardAccum::new();
/// for c in [
///     Card::new(Value::Ace, Suit::Spade),
///     Card::new(Value::King, Suit::Spade),
///     Card::new(Value::Queen, Suit::Spade),
///     Card::new(Value::Jack, Suit::Spade),
///     Card::new(Value::Ten, Suit::Spade),
/// ] {
///     acc.add(c);
/// }
/// // A royal flush, the strongest straight flush.
/// assert_eq!(acc.rank().category(), rs_poker::core::CoreRank::StraightFlush);
/// ```
///
/// # Reusing it for performance
///
/// The struct is [`Copy`] and only 16 bytes, so a partial tally is cheap to
/// clone. Tally the constant cards (hole + already-dealt board) once, then for
/// each candidate runout **copy** the accumulator and `add` only the remaining
/// one or two cards before `rank`. This is the board-enumeration speedup:
/// folding all seven cards from scratch on every runout is the bulk of CFR
/// fast-forward cost, and this avoids it (see `fast_forward_enumerate_showdowns`).
///
/// ```
/// use rs_poker::core::{Card, SevenCardAccum, Suit, Value};
///
/// // Hole + flop: tally once.
/// let mut base = SevenCardAccum::new();
/// for c in [
///     Card::new(Value::Ace, Suit::Spade),
///     Card::new(Value::Ace, Suit::Heart),
///     Card::new(Value::King, Suit::Spade),
///     Card::new(Value::Seven, Suit::Club),
///     Card::new(Value::Two, Suit::Diamond),
/// ] {
///     base.add(c);
/// }
///
/// // Enumerate two turn+river runouts by copying the shared tally.
/// let mut runout_a = base; // Copy
/// runout_a.add(Card::new(Value::Ace, Suit::Club));
/// runout_a.add(Card::new(Value::Ace, Suit::Diamond)); // quad aces
///
/// let mut runout_b = base; // Copy again; `base` is untouched
/// runout_b.add(Card::new(Value::Queen, Suit::Heart));
/// runout_b.add(Card::new(Value::Jack, Suit::Diamond)); // pair of aces
///
/// assert!(runout_a.rank() > runout_b.rank());
/// ```
///
/// # Internals
///
/// A `SevenCardAccum` is exactly the `(key, mask)` pair the evaluator consumes:
/// `key` accumulates each card's rank multiplier and suit-count nibble, `mask`
/// records which cards are present. [`add`](Self::add) folds one card into both
/// with no branching, and [`rank`](Self::rank) hands them to the perfect-hash
/// evaluator in [`crate::core::eval`]. The representation and the tables are
/// documented in full in the `build.rs` header.
#[derive(Clone, Copy)]
pub struct SevenCardAccum {
    key: u64,
    mask: u64,
}

impl Default for SevenCardAccum {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SevenCardAccum {
    #[inline]
    pub fn new() -> Self {
        Self {
            key: eval::DEFAULT_KEY,
            mask: 0,
        }
    }

    /// Fold one card into the tally.
    #[inline]
    pub fn add(&mut self, c: Card) {
        eval::add_card(&mut self.key, &mut self.mask, u8::from(c));
    }

    /// Best 5-card [`Rank`] from the cards folded in so far.
    ///
    /// The perfect-hash tables only cover hands of up to seven cards (the most
    /// a Hold'em player ever holds: two hole cards plus a five-card board), so
    /// ranking more than seven distinct cards yields a meaningless score. A
    /// debug assertion guards against that misuse (e.g. an over-dealt board or
    /// a hand seeded with too many hole cards) rather than silently returning
    /// garbage.
    #[inline]
    pub fn rank(&self) -> Rank {
        debug_assert!(
            self.mask.count_ones() <= 7,
            "hand evaluator supports at most 7 cards, got {}",
            self.mask.count_ones()
        );
        Rank::from_score(eval::evaluate_key(self.key, self.mask))
    }
}

impl std::ops::AddAssign<Card> for SevenCardAccum {
    #[inline]
    fn add_assign(&mut self, c: Card) {
        self.add(c);
    }
}

impl std::ops::Add<Card> for SevenCardAccum {
    type Output = SevenCardAccum;
    #[inline]
    fn add(mut self, c: Card) -> SevenCardAccum {
        SevenCardAccum::add(&mut self, c);
        self
    }
}

// Keep the documented size honest: a compile-time check so the rustdoc figure
// can't silently drift if a field is added.
const _: () = assert!(std::mem::size_of::<SevenCardAccum>() == 16);

/// Accumulate any card iterator and rank it.
fn rank_cards<I: Iterator<Item = Card>>(cards: I) -> Rank {
    let mut acc = SevenCardAccum::new();
    for c in cards {
        acc.add(c);
    }
    acc.rank()
}

impl Rankable for FlatHand {
    fn rank(&self) -> Rank {
        rank_cards(self.iter().copied())
    }
}

impl Rankable for Vec<Card> {
    fn rank(&self) -> Rank {
        rank_cards(self.iter().copied())
    }
}

impl Rankable for [Card] {
    fn rank(&self) -> Rank {
        rank_cards(self.iter().copied())
    }
}

impl Rankable for &[Card] {
    fn rank(&self) -> Rank {
        rank_cards(self.iter().copied())
    }
}

impl Rankable for Hand {
    fn rank(&self) -> Rank {
        rank_cards(self.iter())
    }
}

impl Rankable for CardBitSet {
    fn rank(&self) -> Rank {
        rank_cards(self.into_iter())
    }
}

#[cfg(test)]
pub(crate) mod oracle {
    //! Frozen pre-perfect-hash evaluator, kept only to validate the new one.
    //! Returns an old-style packed u32 (category 0..8 in bits 28.., bitmask
    //! payload below) so ordering and category can be cross-checked.
    const WHEEL: u32 = 0b1_0000_0000_1111;

    fn straight(v: u32) -> Option<u32> {
        let run = v & (v << 1) & (v << 2) & (v << 3) & (v << 4);
        if run != 0 {
            Some(32 - 4 - run.leading_zeros())
        } else if v & WHEEL == WHEEL {
            Some(0)
        } else {
            None
        }
    }
    fn keep_high(v: u32) -> u32 {
        if v == 0 {
            0
        } else {
            1 << (31 - v.leading_zeros())
        }
    }
    fn keep_top(mut v: u32, n: u32) -> u32 {
        while v.count_ones() > n {
            v &= v - 1;
        }
        v
    }

    /// Old-style packed rank from the raw card bitmask (`suit*13+value`).
    /// Category is `0=HighCard..8=StraightFlush` in bits 28.., payload below.
    pub(crate) fn rank_u64(cards: u64) -> u32 {
        let s = [
            (cards & 0x1FFF) as u32,
            ((cards >> 13) & 0x1FFF) as u32,
            ((cards >> 26) & 0x1FFF) as u32,
            ((cards >> 39) & 0x1FFF) as u32,
        ];
        let value_set = s[0] | s[1] | s[2] | s[3];
        let pack = |cat: u32, payload: u32| (cat << 28) | payload;

        let flush = s.iter().find(|m| m.count_ones() >= 5);
        if let Some(&fs) = flush {
            return match straight(fs) {
                Some(r) => pack(8, r),
                None => pack(5, keep_top(fs, 5)),
            };
        }
        let e2 = (s[0] & s[1])
            | (s[0] & s[2])
            | (s[0] & s[3])
            | (s[1] & s[2])
            | (s[1] & s[3])
            | (s[2] & s[3]);
        let e3 = (s[0] & s[1] & s[2])
            | (s[0] & s[1] & s[3])
            | (s[0] & s[2] & s[3])
            | (s[1] & s[2] & s[3]);
        let e4 = s[0] & s[1] & s[2] & s[3];
        let pairs = e2 & !e3;
        let trips = e3 & !e4;
        let quads = e4;
        if quads != 0 {
            pack(7, (quads << 13) | keep_high(value_set ^ quads))
        } else if trips != 0 && trips.count_ones() == 2 {
            let set = keep_high(trips);
            pack(6, (set << 13) | (trips ^ set))
        } else if trips != 0 && pairs != 0 {
            pack(6, (trips << 13) | keep_high(pairs))
        } else if let Some(r) = straight(value_set) {
            pack(4, r)
        } else if trips != 0 {
            pack(3, (trips << 13) | keep_top(value_set ^ trips, 2))
        } else if pairs.count_ones() >= 2 {
            let two = keep_top(pairs, 2);
            pack(2, (two << 13) | keep_high(value_set ^ two))
        } else if pairs != 0 {
            pack(1, (pairs << 13) | keep_top(value_set ^ pairs, 3))
        } else {
            pack(0, keep_top(value_set, 5))
        }
    }

    /// Old-style category 0..8 from the packed value.
    pub(crate) fn category(packed: u32) -> u32 {
        packed >> 28
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Card;
    use crate::core::card::*;

    fn bits_of(cards: &[Card]) -> u64 {
        cards.iter().fold(0u64, |a, c| a | (1u64 << u8::from(*c)))
    }
    // New evaluator over a raw bitmask, for differential comparison.
    fn new_rank(cards: &[Card]) -> Rank {
        let mut acc = SevenCardAccum::new();
        for c in cards {
            acc.add(*c);
        }
        acc.rank()
    }
    // Map old category index (0..8) to CoreRank for agreement checks.
    fn old_core(packed: u32) -> CoreRank {
        match oracle::category(packed) {
            0 => CoreRank::HighCard,
            1 => CoreRank::OnePair,
            2 => CoreRank::TwoPair,
            3 => CoreRank::ThreeOfAKind,
            4 => CoreRank::Straight,
            5 => CoreRank::Flush,
            6 => CoreRank::FullHouse,
            7 => CoreRank::FourOfAKind,
            _ => CoreRank::StraightFlush,
        }
    }

    #[test]
    fn category_agrees_exhaustive_five() {
        let cards: Vec<Card> = (0u8..52).map(Card::from).collect();
        let n = cards.len();
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for d in (c + 1)..n {
                        for e in (d + 1)..n {
                            let hand = [cards[a], cards[b], cards[c], cards[d], cards[e]];
                            let newr = new_rank(&hand);
                            let oldp = oracle::rank_u64(bits_of(&hand));
                            assert_eq!(
                                newr.category(),
                                old_core(oldp),
                                "category mismatch {hand:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn order_isomorphism_exhaustive_five() {
        // Build (old_packed -> new_score). Assert it is a strictly monotonic
        // bijection: each old rank maps to exactly one new score and the orders
        // agree.
        use std::collections::HashMap;
        let cards: Vec<Card> = (0u8..52).map(Card::from).collect();
        let n = cards.len();
        let mut map: HashMap<u32, u16> = HashMap::new();
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for d in (c + 1)..n {
                        for e in (d + 1)..n {
                            let hand = [cards[a], cards[b], cards[c], cards[d], cards[e]];
                            let oldp = oracle::rank_u64(bits_of(&hand));
                            let news = new_rank(&hand).0;
                            if let Some(prev) = map.insert(oldp, news) {
                                assert_eq!(prev, news, "old rank maps to two new scores");
                            }
                        }
                    }
                }
            }
        }
        // Distinct old ranks -> distinct new scores, same order.
        let mut pairs: Vec<(u32, u16)> = map.into_iter().collect();
        pairs.sort_unstable_by_key(|p| p.0);
        for w in pairs.windows(2) {
            assert!(w[0].1 < w[1].1, "ordering not preserved: {w:?}");
        }
    }

    #[test]
    fn differential_random_seven() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        let mut cards: Vec<Card> = (0u8..52).map(Card::from).collect();
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut samples: Vec<(u32, u16)> = Vec::new();
        for _ in 0..200_000 {
            cards.shuffle(&mut rng);
            let hand = &cards[..7];
            let oldp = oracle::rank_u64(bits_of(hand));
            let news = new_rank(hand);
            assert_eq!(
                news.category(),
                old_core(oldp),
                "category mismatch {hand:?}"
            );
            samples.push((oldp, news.0));
        }
        // Any two sampled hands must compare the same way under both rankers.
        for i in 0..samples.len().min(2000) {
            for j in (i + 1)..samples.len().min(2000) {
                let (oa, na) = samples[i];
                let (ob, nb) = samples[j];
                assert_eq!(oa.cmp(&ob), na.cmp(&nb), "pairwise order disagreement");
            }
        }
    }

    #[test]
    fn differential_partial_hands() {
        let cards: Vec<Card> = (0u8..52).map(Card::from).collect();
        for count in [2usize, 3, 4, 6] {
            // Sliding window smoke test; full coverage is the exhaustive_five test.
            for start in 0..(52 - count) {
                let hand = &cards[start..start + count];
                let oldp = oracle::rank_u64(bits_of(hand));
                let news = new_rank(hand);
                assert_eq!(
                    news.category(),
                    old_core(oldp),
                    "category mismatch {hand:?}"
                );
            }
        }
    }

    #[test]
    fn category_ordering_holds() {
        assert!(Rank::HIGH_CARD_MIN < Rank::ONE_PAIR_MIN);
        assert!(Rank::ONE_PAIR_MIN < Rank::TWO_PAIR_MIN);
        assert!(Rank::TWO_PAIR_MIN < Rank::THREE_OF_A_KIND_MIN);
        assert!(Rank::THREE_OF_A_KIND_MIN < Rank::STRAIGHT_MIN);
        assert!(Rank::STRAIGHT_MIN < Rank::FLUSH_MIN);
        assert!(Rank::FLUSH_MIN < Rank::FULL_HOUSE_MIN);
        assert!(Rank::FULL_HOUSE_MIN < Rank::FOUR_OF_A_KIND_MIN);
        assert!(Rank::FOUR_OF_A_KIND_MIN < Rank::STRAIGHT_FLUSH_MIN);
    }

    #[test]
    fn seven_card_accum_size() {
        assert_eq!(std::mem::size_of::<SevenCardAccum>(), 16);
    }

    #[test]
    fn known_hands_have_expected_categories() {
        let sf = FlatHand::new_from_str("AdKdQdJdTd").unwrap();
        assert_eq!(sf.rank().category(), CoreRank::StraightFlush);
        let quads = FlatHand::new_from_str("AsAhAdAcKs").unwrap();
        assert_eq!(quads.rank().category(), CoreRank::FourOfAKind);
        let wheel = FlatHand::new_from_str("Ad2c3s4h5d").unwrap();
        assert_eq!(wheel.rank().category(), CoreRank::Straight);
    }

    // CoreRank conversion tests (do not depend on Rank internals).
    #[test]
    fn test_core_rank_from_categories() {
        assert_eq!(CoreRank::HighCard, Rank::HIGH_CARD_MIN.category());
        assert_eq!(CoreRank::OnePair, Rank::ONE_PAIR_MIN.category());
        assert_eq!(CoreRank::TwoPair, Rank::TWO_PAIR_MIN.category());
        assert_eq!(CoreRank::ThreeOfAKind, Rank::THREE_OF_A_KIND_MIN.category());
        assert_eq!(CoreRank::Straight, Rank::STRAIGHT_MIN.category());
        assert_eq!(CoreRank::Flush, Rank::FLUSH_MIN.category());
        assert_eq!(CoreRank::FullHouse, Rank::FULL_HOUSE_MIN.category());
        assert_eq!(CoreRank::FourOfAKind, Rank::FOUR_OF_A_KIND_MIN.category());
        assert_eq!(CoreRank::StraightFlush, Rank::STRAIGHT_FLUSH_MIN.category());
    }

    #[test]
    fn test_core_rank_into() {
        let r: CoreRank = Rank::FLUSH_MIN.into();
        assert_eq!(r, CoreRank::Flush);
    }

    #[test]
    fn test_core_rank_ordering() {
        assert!(CoreRank::HighCard < CoreRank::OnePair);
        assert!(CoreRank::OnePair < CoreRank::TwoPair);
        assert!(CoreRank::TwoPair < CoreRank::ThreeOfAKind);
        assert!(CoreRank::ThreeOfAKind < CoreRank::Straight);
        assert!(CoreRank::Straight < CoreRank::Flush);
        assert!(CoreRank::Flush < CoreRank::FullHouse);
        assert!(CoreRank::FullHouse < CoreRank::FourOfAKind);
        assert!(CoreRank::FourOfAKind < CoreRank::StraightFlush);
    }

    #[test]
    fn test_core_rank_display() {
        assert_eq!(CoreRank::HighCard.to_string(), "High Card");
        assert_eq!(CoreRank::OnePair.to_string(), "One Pair");
        assert_eq!(CoreRank::TwoPair.to_string(), "Two Pair");
        assert_eq!(CoreRank::ThreeOfAKind.to_string(), "Three of a Kind");
        assert_eq!(CoreRank::Straight.to_string(), "Straight");
        assert_eq!(CoreRank::Flush.to_string(), "Flush");
        assert_eq!(CoreRank::FullHouse.to_string(), "Full House");
        assert_eq!(CoreRank::FourOfAKind.to_string(), "Four of a Kind");
        assert_eq!(CoreRank::StraightFlush.to_string(), "Straight Flush");
    }

    // Ordering within a category, via concrete hands.
    #[test]
    fn test_rank_ordering_within_same_type() {
        let pair_aces = FlatHand::new_from_str("AsAhKdQcJs").unwrap();
        let pair_kings = FlatHand::new_from_str("KsKhAdQcJs").unwrap();
        assert!(pair_aces.rank() > pair_kings.rank());

        let two_pair_ak = FlatHand::new_from_str("AsAhKdKcJs").unwrap();
        let two_pair_aq = FlatHand::new_from_str("AsAhQdQcKs").unwrap();
        assert!(two_pair_ak.rank() > two_pair_aq.rank());

        let trips_aces = FlatHand::new_from_str("AsAhAdKcJs").unwrap();
        let trips_kings = FlatHand::new_from_str("KsKhKdAcJs").unwrap();
        assert!(trips_aces.rank() > trips_kings.rank());
    }

    #[test]
    fn test_rankable_vec_and_slice() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
        ];
        assert_eq!(cards.rank().category(), CoreRank::StraightFlush);
        let slice: &[Card] = &cards;
        assert_eq!(slice.rank().category(), CoreRank::StraightFlush);
        assert_eq!(cards[..].rank().category(), CoreRank::StraightFlush);
    }

    #[test]
    fn test_wheel_straight_detection() {
        let wheel = FlatHand::new_from_str("Ad2c3s4h5d").unwrap();
        assert_eq!(wheel.rank().category(), CoreRank::Straight);
        // Compares below the lowest non-wheel straight (six-high).
        let six_high = FlatHand::new_from_str("2c3s4h5d6c").unwrap();
        assert!(wheel.rank() < six_high.rank());

        let not_wheel = FlatHand::new_from_str("Ad2c3s4h6d").unwrap();
        assert_eq!(not_wheel.rank().category(), CoreRank::HighCard);
        let almost_wheel = FlatHand::new_from_str("Ad2c3s4h6c").unwrap();
        assert_eq!(almost_wheel.rank().category(), CoreRank::HighCard);
    }

    #[test]
    fn test_seven_card_categories() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
            Card::new(Value::Nine, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
        ];
        assert_eq!(cards.rank().category(), CoreRank::StraightFlush);
    }

    #[test]
    fn seven_card_accum_matches_rank_best_of() {
        // One 7-card hand per rank category; incremental add must equal the
        // all-at-once FlatHand ranking.
        for s in [
            "Ad8h9cTc5c2s7d", // high card
            "AdAc9d8cTs2h3s", // one pair
            "AdAc9d8cTs8s3s", // two pair
            "AdAcAs8cTs2h3s", // three of a kind
            "2c3s4h5s6d8cKh", // straight
            "Ad8d9dTd5d2h3s", // flush
            "AdAc9d9c9s2h3s", // full house
            "AdAcAsAh8cTs2h", // four of a kind
            "AdKdQdJdTd9d8d", // straight flush
        ] {
            let hand = FlatHand::new_from_str(s).unwrap();
            let mut acc = SevenCardAccum::new();
            for c in hand.iter() {
                acc.add(*c);
            }
            assert_eq!(acc.rank(), hand.rank(), "mismatch for {s}");
        }
    }

    #[test]
    fn seven_card_accum_order_independent() {
        let hand = FlatHand::new_from_str("2s2h2d2c8d8sKd").unwrap();
        let cards: Vec<Card> = hand.iter().copied().collect();

        let mut forward = SevenCardAccum::new();
        for c in &cards {
            forward.add(*c);
        }
        let mut backward = SevenCardAccum::new();
        for c in cards.iter().rev() {
            backward.add(*c);
        }
        assert_eq!(forward.rank(), backward.rank());
        assert_eq!(forward.rank(), hand.rank());
    }

    #[test]
    fn seven_card_accum_add_operators() {
        let base = FlatHand::new_from_str("AsAhKs7c2d").unwrap();
        let mut via_assign = SevenCardAccum::new();
        for c in base.iter() {
            via_assign += *c;
        }
        let via_add = base.iter().fold(SevenCardAccum::new(), |acc, c| acc + *c);
        assert_eq!(via_assign.rank(), via_add.rank());
        assert_eq!(via_assign.rank(), base.rank());
    }

    #[test]
    fn seven_card_accum_copy_reuse_equals_from_scratch() {
        // Tally a shared 5-card base once, then COPY it for two different
        // 2-card extensions, the reuse pattern that makes enumeration fast.
        let base_hand = FlatHand::new_from_str("AsAhKs7c2d").unwrap();
        let mut base = SevenCardAccum::new();
        for c in base_hand.iter() {
            base.add(*c);
        }

        let mut quads = base; // Copy; `base` is untouched.
        quads.add(Card::new(Value::Ace, Suit::Club));
        quads.add(Card::new(Value::Ace, Suit::Diamond));
        let quads_scratch = FlatHand::new_from_str("AsAhKs7c2dAcAd").unwrap();
        assert_eq!(quads.rank(), quads_scratch.rank());

        let mut pair = base; // Copy again from the same untouched base.
        pair.add(Card::new(Value::Queen, Suit::Heart));
        pair.add(Card::new(Value::Jack, Suit::Diamond));
        let pair_scratch = FlatHand::new_from_str("AsAhKs7c2dQhJd").unwrap();
        assert_eq!(pair.rank(), pair_scratch.rank());

        // Four-of-a-kind outranks one pair.
        assert!(quads.rank() > pair.rank());
    }
}
