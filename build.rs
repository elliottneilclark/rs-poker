//! Build script: generates the perfect-hash hand-evaluator tables into
//! `$OUT_DIR/eval_tables.rs`, which `src/core/eval.rs` then `include!`s.
//!
//! The algorithm is reimplemented from zekyll's OMPEval (MIT). This file is the
//! "compiler": it does all the expensive combinatorial work once, at build time,
//! and bakes the answers into flat arrays. At runtime, ranking a hand is then
//! just a couple of array loads and no branches worth mentioning.
//!
//!
//! # 1. What a "rank" is
//!
//! Every poker hand can be scored by a single 16-bit number. Higher is better.
//! We pack two fields into it:
//!
//! ```text
//!   bit:  15 .. 12 | 11 ............ 0
//!         category  | subrank
//!         (1..=9)   | (0..=4095)
//! ```
//!
//! `category` is the hand class, ordered worst to best:
//!
//! ```text
//!   1 HIGH_CARD   4 TRIPS      7 FULL_HOUSE
//!   2 PAIR        5 STRAIGHT   8 QUADS
//!   3 TWO_PAIR    6 FLUSH      9 STRAIGHT_FLUSH
//! ```
//!
//! `subrank` breaks ties inside a category (which two pair, which kicker, etc.).
//! Because category sits in the high bits, comparing two scores as plain
//! integers gives the correct poker ordering for free.
//!
//! There are exactly 7462 distinct five-card hands once you ignore suits that do
//! not form a flush. Our generated tables collapse every possible hand onto one
//! of those 7462 scores; the build asserts that count as a self-check.
//!
//! Where does 7462 come from? There are C(52,5) = 2,598,960 distinct five-card
//! deals, but most are equivalent for ranking: clubs are interchangeable with
//! spades unless five of them make a flush, and the order you hold cards in
//! never matters. Counting the genuinely distinct VALUES, by category:
//!
//! ```text
//!   category          count   how it is counted (13 ranks)
//!   ---------------   -----   ----------------------------------------------
//!   Straight flush       10   one per high card, A-5 (wheel) up to 10-A
//!   Four of a kind      156   13 quad ranks * 12 kicker ranks
//!   Full house          156   13 trips ranks * 12 pair ranks
//!   Flush             1,277   C(13,5) rank patterns - 10 straight patterns
//!   Straight             10   same 10 high cards as straight flush
//!   Three of a kind     858   13 trips ranks * C(12,2) kicker pairs
//!   Two pair            858   C(13,2) pair ranks * 11 kicker ranks
//!   One pair          2,860   13 pair ranks * C(12,3) kicker triples
//!   High card         1,277   same C(13,5) - 10 patterns, but off-suit
//!   ---------------   -----
//!   TOTAL             7,462
//! ```
//!
//! Flush and high card share the same 1,277 rank patterns: any five distinct
//! ranks that are not a straight. The only difference is whether they are
//! one-suited, and that is exactly the "suits matter only for a flush" rule. The
//! straight and straight-flush rows likewise share their 10 patterns. Sum the
//! column and you get 7,462; `assert_7462_classes` reconstructs this set from the
//! generated tables and checks the size.
//!
//!
//! # 2. The (key, mask) representation of a hand
//!
//! A hand is accumulated into two `u64` words as cards are dealt in. Both are
//! purely additive: dealing a card does `key += CARDS_KEY[c]` and
//! `mask |= CARDS_MASK[c]`, with no per-card branching. (See `eval.rs` for the
//! runtime side; this script generates `CARDS_KEY`, `CARDS_MASK`, and the
//! starting `DEFAULT_KEY`.)
//!
//! A card index is `suit * 13 + value`, so 0..=51:
//!
//! ```text
//!   value: 0=2 1=3 2=4 ... 8=10 9=J 10=Q 11=K 12=A
//!   suit : 0,1,2,3
//!   index = suit*13 + value
//! ```
//!
//! ## 2a. mask: one bit per card
//!
//! `mask` is the obvious thing: bit `index` is set when that exact card is
//! present. Its only job is the flush path. The 13 bits for one suit can be
//! sliced out with `(mask >> (13 * suit)) & 0x1FFF` to get that suit's rank
//! pattern.
//!
//! ```text
//!   mask bit layout (52 used bits):
//!     [ suit3 13 bits | suit2 13 bits | suit1 13 bits | suit0 13 bits ]
//!      63..           ...                                          ..0
//! ```
//!
//! ## 2b. key: rank fingerprint (low) + suit counters (high)
//!
//! `key` does double duty across two disjoint regions of the 64 bits:
//!
//! ```text
//!   key bit layout:
//!     63    60 59    56 55    52 51    48 47 ........................ 0
//!     [ suit3 ][ suit2 ][ suit1 ][ suit0 ][   additive rank key (32) ]
//!       nibble   nibble   nibble   nibble
//! ```
//!
//! Low 32 bits: the additive rank key. This ignores suit entirely and is a
//! single 32-bit fingerprint of the hand's rank histogram (the count of each
//! rank held). It is NOT a bit-packed array of counts; it is an arithmetic sum
//! that happens to be unique per histogram. Section 3 is entirely about this.
//!
//! High 16 bits: four 4-bit nibbles, one per suit, each a running count of how
//! many cards of that suit have been added. This is how we detect a flush
//! without a loop.
//!
//! ## 2c. The flush-bias trick
//!
//! Each suit nibble is pre-loaded with 3 (that is what `DEFAULT_KEY = 0x3333`
//! in the high nibbles means). Each card of a suit adds 1 to its nibble. So:
//!
//! ```text
//!   cards of suit:  0    1    2    3    4    5    6    7
//!   nibble value :  3    4    5    6    7    8    9   10
//!   binary       : 0011 0100 0101 0110 0111 1000 1001 1010
//!                                            ^ the 0x8 bit flips at exactly 5
//! ```
//!
//! A flush needs 5+ cards of one suit, and 3 + 5 = 8 = `0b1000`. So the top bit
//! of a nibble (`0x8`) is set if and only if that suit has reached a flush.
//! Testing `key & (0x8888 << 48)` tells us in one AND whether any flush exists,
//! and which suit. The bias of 3 also guarantees the nibble never overflows: the
//! most cards of one suit in a 7-card hand is 7, giving 10, still inside 4 bits.
//!
//!
//! # 3. The additive rank key (low 32 bits)
//!
//! Forget suits for this whole section.
//!
//! ## 3a. The rank histogram (the concept)
//!
//! A hand's non-flush strength is fully determined by its rank histogram: how
//! many of each rank you hold. Write it as a 13-wide vector `counts[0..13]`,
//! where `counts[r]` is the number of cards of rank `r` (each 0..=4, total
//! 0..=7). Rank indices are `0=2, 1=3, ... 9=J, 10=Q, 11=K, 12=A`. For example,
//! the five cards K K A Q 7 give:
//!
//! ```text
//!   rank :  2  3  4  5  6  7  8  9  T  J  Q  K  A
//!   index:  0  1  2  3  4  5  6  7  8  9 10 11 12
//!   count:  0  0  0  1  0  0  0  0  0  0  1  2  1
//!                       (7)              (Q)(K)(A)
//! ```
//!
//! Suit is gone: K-of-clubs and K-of-spades both just bump `counts[11]`. Two
//! different deals with the same histogram are the same non-flush hand, so the
//! histogram is the thing we want to look up. The trouble is it is 13 numbers; we
//! want one small integer to index an array with.
//!
//! ## 3b. Collapsing the histogram to one number (the key)
//!
//! We do NOT store the 13 counts as bit-fields. Instead we give each rank a magic
//! 32-bit multiplier `RANKS[r]` and collapse the whole vector to a single sum:
//!
//! ```text
//!   rank_key = sum over r of  RANKS[r] * counts[r]    (wrapping in u32)
//! ```
//!
//! This is why the layout is "additive": each card of rank `r` just adds the
//! constant `RANKS[r]` into the accumulator (that is the low 32 bits of
//! `CARDS_KEY`), so building the key needs no branches and no per-rank fields,
//! only `key += CARDS_KEY[card]` per card. The same K K A Q 7:
//!
//! ```text
//!   1 * RANKS[ 5]  (the 7)   = 0x00176005
//!   1 * RANKS[10]  (the Q)   = 0x0048c0e4
//!   2 * RANKS[11]  (two Ks)  = 0x0091e422
//!   1 * RANKS[12]  (the A)   = 0x00494493
//!   ------------------------------------- wrapping sum
//!   rank_key                 = 0x013b499e
//! ```
//!
//! The `rank_key` is a fingerprint of the histogram: you cannot read the
//! individual counts back out of it, but you do not need to. All that matters is
//! that the same histogram always yields the same number, and different
//! histograms yield different numbers.
//!
//! ## 3c. Why the magic constants work
//!
//! That second property is a strong claim: the `RANKS` constants (lifted from
//! OMPEval) are hand-picked so that every reachable histogram produces a DISTINCT
//! `rank_key`, with no collisions across the wrapping add. That is exactly what
//! makes the sum usable as a lookup key at all. The build does not take it on
//! faith: `assert_rank_keys_distinct` enumerates all reachable histograms and
//! checks that their keys are pairwise distinct. If a constant were ever wrong,
//! the build fails.
//!
//! These keys are sparse and large (spread across a 25-bit space, e.g. our
//! example sits at 0x013b499e), so we cannot index an array with them directly
//! without wasting gigabytes. Section 5 shows how we compress them.
//!
//!
//! # 4. From histogram to score: classify, then densify
//!
//! Two steps turn a histogram into a final 16-bit score.
//!
//! `classify` (section: the classifier) walks a fixed ladder of hand types and
//! returns `(category, payload)`. The `payload` is an ordering number: within a
//! category, a larger payload means a stronger hand, but the payloads are NOT
//! contiguous and their exact bit widths do not matter.
//!
//! `densify` then squashes each category's set of distinct payloads down to a
//! contiguous range of subranks 1, 2, 3, ... ordered by payload. Two histograms
//! that reach the same best five cards (e.g. the same flush makeable from
//! different six-card holdings) share a payload, hence share a subrank. After
//! densifying, `score = (category << 12) | subrank`.
//!
//!
//! # 5. Compressing the keys: a row-displacement perfect hash
//!
//! We have ~7000 live `rank_key` values scattered across a 25-bit space, and we
//! want an array small enough to fit in cache where `array[h(key)] == score`
//! with NO collisions and NO search at runtime. That is a "perfect hash".
//!
//! The construction used here is "first-fit row displacement". Picture the keys
//! laid out in a grid. Split each key into a high part (the "row") and a low part
//! (the "column"):
//!
//! ```text
//!   row    = key >> ROW_SHIFT        (top bits; which row)
//!   column = key & (ROW_WIDTH - 1)   (low ROW_SHIFT bits; offset within row)
//! ```
//!
//! Now imagine a 2-D grid, one populated row per distinct `row` value, with the
//! keys of that row sitting at their `column` positions:
//!
//! ```text
//!            column ->  0    1    2   ...            ROW_WIDTH-1
//!   row 0           [ .    K    .    .   K    .  ...            ]
//!   row 1           [ K    .    .    K   .    .  ...            ]
//!   row 2           [ .    .    K    .   .    K  ...            ]
//! ```
//!
//! We want to flatten this into one 1-D array (`LOOKUP`) with no two `K`s
//! landing on the same cell. We do it by sliding each whole row left or right by
//! a per-row amount, the "displacement" or `offset`, until that row's keys drop
//! only into still-empty cells of the flat array:
//!
//! ```text
//!   slot(key) = (key + ROW_OFFSETS[row]) wrapped into [0, LOOKUP_LEN)
//! ```
//!
//! Because every key in a row shares the same `row`, adding `offset` shifts the
//! whole row rigidly and preserves the spacing between its keys. We place the
//! densest rows first (they are the hardest to fit), and for each row we scan
//! `offset = 0, 1, 2, ...` taking the first one that does not collide -- "first
//! fit". `build_perfect_hash` does exactly this and then verifies that every key
//! reads its own score back out.
//!
//! At runtime, ranking a non-flush hand is therefore: take `rank_key`, look up
//! `ROW_OFFSETS[rank_key >> ROW_SHIFT]`, add it, and load `LOOKUP[that slot]`.
//!
//!
//! # 6. The flush path (separate, simpler)
//!
//! Flushes cannot use the rank-histogram key, because a flush cares which suit
//! and the histogram threw suits away. But a flush is simpler: it is decided by
//! the 13-bit rank pattern of the one flushed suit, which we already have in
//! `mask`. So we precompute a direct table `FLUSH_LOOKUP[pattern] = score` for
//! all 8192 patterns. `build_flush_table` fills it, scoring straight flushes
//! above ordinary flushes. No perfect hash needed; the pattern IS the index.
//!
//!
//! # 7. What this script emits
//!
//! Into `$OUT_DIR/eval_tables.rs`:
//!   - sizing consts: LOOKUP_LEN, FLUSH_LEN, ROW_SHIFT, ROW_OFFSETS_LEN, DEFAULT_KEY
//!   - LOOKUP        : non-flush perfect-hash table (score per slot)
//!   - FLUSH_LOOKUP  : flush table (score per 13-bit suit pattern)
//!   - ROW_OFFSETS   : per-row displacements for the perfect hash
//!   - CARDS_KEY     : per-card key contribution (rank multiplier + suit nibble)
//!   - CARDS_MASK    : per-card mask bit
//!
//! The five self-check functions (`assert_*`) guard the invariants the runtime
//! depends on. Any violation fails the build rather than shipping a wrong table.

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

/// Per-rank key multipliers (OMPEval, MIT), indexed by value 0..=12.
///
/// These are the magic numbers from section 3: `rank_key` is the wrapping sum of
/// `RANKS[r] * counts[r]`, and they are chosen so distinct reachable histograms
/// never collide. `assert_rank_keys_distinct` proves that at build time.
const RANKS: [u32; 13] = [
    0x2000, 0x8001, 0x11000, 0x3a000, 0x91000, 0x176005, 0x366000, 0x41a013, 0x47802e, 0x479068,
    0x48c0e4, 0x48f211, 0x494493,
];

/// Length of the non-flush perfect-hash table. A power of two so the runtime can
/// index it with a mask instead of a modulo. Large enough that first-fit row
/// displacement always finds room.
const LOOKUP_LEN: usize = 1 << 17;

/// Length of the flush table: one entry per 13-bit suit pattern.
const FLUSH_LEN: usize = 8192;

/// How many low bits of a `rank_key` form the "column"; the remaining high bits
/// form the "row". See the row-displacement diagram in section 5.
const ROW_SHIFT: u32 = 12;

/// Number of rows: `rank_key >> ROW_SHIFT` must land in `0..ROW_OFFSETS_LEN`.
const ROW_OFFSETS_LEN: usize = 8192;

// Hand categories, the high nibble of every score. See section 1.
const HIGH_CARD: u32 = 1;
const PAIR: u32 = 2;
const TWO_PAIR: u32 = 3;
const TRIPS: u32 = 4;
const STRAIGHT: u32 = 5;
const FLUSH: u32 = 6;
const FULL_HOUSE: u32 = 7;
const QUADS: u32 = 8;
const STRAIGHT_FLUSH: u32 = 9;

/// The non-flush categories, in the order their tables are densified.
const NONFLUSH_CATS: [u32; 7] = [
    HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FULL_HOUSE, QUADS,
];

/// Rank-mask of the wheel straight A-2-3-4-5: aces low. Bit 12 (ace) plus bits
/// 0..=3 (five, four, three, two). Used by `straight`.
const WHEEL: u32 = 0b1_0000_0000_1111;

// ============================================================================
// Shared types
// ============================================================================

/// One classified hand source: `(source_key, category, payload)`.
///
/// `source_key` is a `rank_key` for non-flush entries or a 13-bit suit pattern
/// for flush entries; `densify` and the classifier only ever read `category` and
/// `payload`, so the same triple serves both paths.
type ClassEntry = (u32, u32, u32);

/// Map from a histogram's `rank_key` to its packed 16-bit score.
type ValueMap = HashMap<u32, u16>;

// ============================================================================
// Rank-mask primitives
//
// A "rank mask" is a 13-bit value: bit `r` set means rank `r` is present. These
// three helpers operate on rank masks and are the building blocks of both the
// classifier and the flush table.
// ============================================================================

/// Detect the best straight in a 13-bit rank mask.
///
/// Returns `Some(subrank)` where a larger subrank is a higher straight, or
/// `None` if there is no straight. The wheel (A-2-3-4-5) is the lowest straight
/// and maps to `Some(0)`.
///
/// The five-in-a-row test is branch-free. `v & (v<<1) & (v<<2) & (v<<3) & (v<<4)`
/// keeps a bit only where five consecutive ranks are all present:
///
/// ```text
///   v        : ..0 0 1 1 1 1 1 0..   (ranks r..r+4 present)
///   v<<1     : ..0 1 1 1 1 1 0 0..
///   v<<2     : ..1 1 1 1 1 0 0 0..
///   v<<3..4  : (shifted further)
///   AND      : ..0 0 0 0 1 0 0 0..   one surviving bit at the top of the run
/// ```
///
/// The surviving bit's position identifies the straight's high card. The wheel
/// is checked separately because its ace wraps from bit 12 down to bit 0.
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

/// Keep only the highest `n` set bits of a rank mask, clearing the rest.
///
/// Used to drop kickers that do not count, e.g. a high-card hand keeps its top 5
/// ranks. Repeatedly clears the lowest set bit (`v &= v - 1`) until `n` remain.
fn keep_top(mut v: u32, n: u32) -> u32 {
    while v.count_ones() > n {
        v &= v - 1;
    }
    v
}

/// Keep only the single highest set bit of a rank mask (0 stays 0).
fn keep_high(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        1 << (31 - v.leading_zeros())
    }
}

// ============================================================================
// The non-flush classifier
// ============================================================================

/// Classify a rank histogram into `(category, payload)`.
///
/// `counts[r]` is how many cards of rank `r` the hand holds. The return is the
/// hand `category` (section 1) and an ordering `payload`: within one category a
/// larger payload is a stronger hand. Payloads are not contiguous and their bit
/// widths are unimportant; `densify` compresses them to subranks afterwards.
///
/// This mirrors the runtime non-flush ladder in `rank_u64`. The ladder is walked
/// strongest structural feature first (quads, then full house, ...). The
/// `kept << 13 | kickers` shape simply stacks the defining ranks above the
/// kicker ranks so the payload sorts correctly.
fn classify(counts: &[u8; 13]) -> (u32, u32) {
    // Derive four rank masks from the histogram in one pass.
    let mut value_set = 0u32; // ranks present at all
    let mut pairs = 0u32; // ranks held exactly twice
    let mut trips = 0u32; // ranks held exactly three times
    let mut quads = 0u32; // ranks held four+ times
    for (r, &c) in counts.iter().enumerate() {
        let bit = 1u32 << r;
        if c >= 1 {
            value_set |= bit;
        }
        if c == 2 {
            pairs |= bit;
        }
        if c == 3 {
            trips |= bit;
        }
        if c >= 4 {
            quads |= bit;
        }
    }

    if quads != 0 {
        // Quads plus the single best remaining kicker.
        let high = keep_high(value_set ^ quads);
        (QUADS, (quads << 13) | high)
    } else if trips != 0 && trips.count_ones() == 2 {
        // Two trips (only reachable with 6+ cards): the higher fills the trips
        // slot, the lower acts as the pair.
        let set = keep_high(trips);
        let pair = trips ^ set;
        (FULL_HOUSE, (set << 13) | pair)
    } else if trips != 0 && pairs != 0 {
        // Trips plus a pair: pick the best pair.
        let pair = keep_high(pairs);
        (FULL_HOUSE, (trips << 13) | pair)
    } else if let Some(s) = straight(value_set) {
        (STRAIGHT, s)
    } else if trips != 0 {
        // Trips plus its two best kickers.
        let low = keep_top(value_set ^ trips, 2);
        (TRIPS, (trips << 13) | low)
    } else if pairs.count_ones() >= 2 {
        // Two pair: the best two pairs plus the best remaining kicker.
        let two = keep_top(pairs, 2);
        let low = keep_high(value_set ^ two);
        (TWO_PAIR, (two << 13) | low)
    } else if pairs != 0 {
        // One pair plus its three best kickers.
        let low = keep_top(value_set ^ pairs, 3);
        (PAIR, (pairs << 13) | low)
    } else {
        // High card: the best five ranks.
        (HIGH_CARD, keep_top(value_set, 5))
    }
}

// ============================================================================
// Histogram enumeration
// ============================================================================

/// The additive rank key of a histogram: the wrapping sum of
/// `RANKS[r] * counts[r]`. See section 3.
fn rank_key(counts: &[u8; 13]) -> u32 {
    (0..13)
        .map(|r| RANKS[r].wrapping_mul(counts[r] as u32))
        .fold(0u32, u32::wrapping_add)
}

/// Visit every reachable rank histogram: each rank 0..=4, total cards 0..=7.
///
/// This is the full domain of the non-flush evaluator. A 7-card hold'em hand can
/// hold at most four of a rank and seven cards overall, so these bounds cover
/// every histogram the runtime can ever see.
fn enumerate_counts(counts: &mut [u8; 13], rank: usize, total: u8, f: &mut impl FnMut(&[u8; 13])) {
    if rank == 13 {
        f(counts);
        return;
    }
    let max = std::cmp::min(4, 7 - total);
    for c in 0..=max {
        counts[rank] = c;
        enumerate_counts(counts, rank + 1, total + c, f);
    }
    counts[rank] = 0;
}

/// Like `enumerate_counts`, but only visits histograms whose total is exactly
/// `remaining`. Used by the 7462-class self-check to enumerate five-card hands.
fn enumerate_counts_exact(
    counts: &mut [u8; 13],
    rank: usize,
    remaining: u8,
    f: &mut impl FnMut(&[u8; 13]),
) {
    if rank == 13 {
        if remaining == 0 {
            f(counts);
        }
        return;
    }
    let max = std::cmp::min(4, remaining);
    for c in 0..=max {
        counts[rank] = c;
        enumerate_counts_exact(counts, rank + 1, remaining - c, f);
    }
    counts[rank] = 0;
}

// ============================================================================
// Densification: payloads -> contiguous subranks
// ============================================================================

/// Map each distinct payload in `cat` to a contiguous subrank, ordered by
/// payload.
///
/// `entries` is `(rank_key, category, payload)` triples. The smallest payload in
/// the category becomes subrank 1, the next becomes 2, and so on. Equal payloads
/// (the same best-five reached different ways) collapse to one subrank. The
/// returned map is `payload -> subrank` for that one category.
fn densify(cat: u32, entries: &[ClassEntry]) -> HashMap<u32, u32> {
    let mut payloads: Vec<u32> = entries.iter().filter(|e| e.1 == cat).map(|e| e.2).collect();
    payloads.sort_unstable();
    payloads.dedup();
    let mut map = HashMap::new();
    for (i, p) in payloads.iter().enumerate() {
        let subrank = (i + 1) as u32;
        assert!(subrank < 4096, "subrank overflow in category {cat}");
        map.insert(*p, subrank);
    }
    map
}

// ============================================================================
// The non-flush value map: histogram key -> 16-bit score
// ============================================================================

/// Build `rank_key -> score` for every reachable non-flush histogram.
///
/// Enumerates all histograms, classifies each into `(category, payload)`, then
/// densifies payloads per category into subranks and packs `(category, subrank)`
/// into the final 16-bit score. Different histograms with the same `rank_key`
/// always classify to the same score (that is what the distinctness of keys
/// buys us), so inserting by key is unambiguous.
///
/// Also returns the raw `(rank_key, category, payload)` entries, which the
/// distinctness self-check reuses.
fn build_value_map() -> (ValueMap, Vec<ClassEntry>) {
    let mut entries: Vec<ClassEntry> = Vec::new();
    let mut counts = [0u8; 13];
    enumerate_counts(&mut counts, 0, 0, &mut |counts| {
        let (cat, payload) = classify(counts);
        entries.push((rank_key(counts), cat, payload));
    });

    let mut value_of: ValueMap = HashMap::new();
    for &cat in &NONFLUSH_CATS {
        let sub = densify(cat, &entries);
        for e in entries.iter().filter(|e| e.1 == cat) {
            let score = ((cat << 12) | sub[&e.2]) as u16;
            value_of.insert(e.0, score);
        }
    }
    (value_of, entries)
}

// ============================================================================
// The row-displacement perfect hash (section 5)
// ============================================================================

/// Slot a key lands in given its row's displacement, wrapped to `LOOKUP_LEN`.
///
/// This is the single source of the addressing arithmetic; `perf_hash` in
/// `eval.rs` mirrors it exactly. `LOOKUP_LEN` is a power of two, so the wrap is a
/// mask.
fn perf_slot(k: u32, offset: u32) -> usize {
    k.wrapping_add(offset) as usize & (LOOKUP_LEN - 1)
}

/// Place every live key into a collision-free `LOOKUP` table via first-fit row
/// displacement, returning `(LOOKUP, ROW_OFFSETS)`.
///
/// Steps, matching the diagram in section 5:
///   1. Bucket keys by `row = key >> ROW_SHIFT`.
///   2. Process rows densest first (hardest to place).
///   3. For each row, scan `offset = 0, 1, 2, ...` and take the first that drops
///      all of the row's keys into still-empty slots; mark those slots used.
///   4. Fill `LOOKUP[perf_slot(key, offset_of_its_row)] = score`.
///
/// Unused slots stay 0, which decodes as an impossible score, so a stray lookup
/// is detectably wrong rather than silently plausible.
fn build_perfect_hash(value_of: &ValueMap) -> (Vec<u16>, Vec<u32>) {
    // 1. Bucket the live keys by row.
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); ROW_OFFSETS_LEN];
    for &k in value_of.keys() {
        let row = (k >> ROW_SHIFT) as usize;
        assert!(row < ROW_OFFSETS_LEN, "row index out of range");
        rows[row].push(k);
    }

    // 2. Densest rows first: a packed array gets harder to fit as it fills, so
    //    place the demanding rows while there is still room.
    let mut order: Vec<usize> = (0..ROW_OFFSETS_LEN).collect();
    order.sort_by_key(|&r| std::cmp::Reverse(rows[r].len()));

    // 3. First-fit each row.
    let mut used = vec![false; LOOKUP_LEN];
    let mut offsets = vec![0u32; ROW_OFFSETS_LEN];
    for &row in &order {
        if rows[row].is_empty() {
            continue;
        }
        let mut offset: u32 = 0;
        loop {
            let fits = rows[row].iter().all(|&k| !used[perf_slot(k, offset)]);
            if fits {
                break;
            }
            offset = offset
                .checked_add(1)
                .expect("row-displacement offset overflow");
            assert!(
                (offset as usize) < LOOKUP_LEN,
                "no displacement fits row {row}; increase LOOKUP_LEN"
            );
        }
        for &k in &rows[row] {
            used[perf_slot(k, offset)] = true;
        }
        offsets[row] = offset;
    }

    // 4. Emit the scores into their slots.
    let mut lookup = vec![0u16; LOOKUP_LEN];
    for (&k, &score) in value_of {
        lookup[perf_slot(k, offsets[(k >> ROW_SHIFT) as usize])] = score;
    }
    (lookup, offsets)
}

// ============================================================================
// The flush table (section 6)
// ============================================================================

/// Build `FLUSH_LOOKUP`: a direct table from a 13-bit suit pattern to its score.
///
/// Only patterns with 5+ bits are flushes; the rest stay 0 and are never queried
/// (the runtime only consults this table once a flush is known to exist). A
/// pattern that is also five-in-a-row scores as a straight flush, otherwise as a
/// plain flush keeping its top five ranks. Both categories are densified just
/// like the non-flush ones.
fn build_flush_table() -> Vec<u16> {
    // Score every 5+ bit pattern into (pattern, category, payload).
    let mut flush_cat: Vec<ClassEntry> = Vec::new();
    for pattern in 0u32..(FLUSH_LEN as u32) {
        if pattern.count_ones() < 5 {
            continue;
        }
        let (cat, payload) = match straight(pattern) {
            Some(s) => (STRAIGHT_FLUSH, s),
            None => (FLUSH, keep_top(pattern, 5)),
        };
        flush_cat.push((pattern, cat, payload));
    }

    let flush_sub = densify(FLUSH, &flush_cat);
    let sf_sub = densify(STRAIGHT_FLUSH, &flush_cat);

    let mut flush_lookup = vec![0u16; FLUSH_LEN];
    for &(pattern, cat, payload) in &flush_cat {
        let sub = if cat == FLUSH {
            flush_sub[&payload]
        } else {
            sf_sub[&payload]
        };
        flush_lookup[pattern as usize] = ((cat << 12) | sub) as u16;
    }
    flush_lookup
}

// ============================================================================
// Per-card key and mask contributions (section 2)
// ============================================================================

/// Build the per-card contribution tables and the starting key.
///
/// Returns `(CARDS_KEY, CARDS_MASK, DEFAULT_KEY)`:
///   - `CARDS_KEY[i]` adds the card's rank multiplier into the low 32 bits and
///     +1 into its suit nibble at bit `48 + 4*suit`.
///   - `CARDS_MASK[i]` is just `1 << i`.
///   - `DEFAULT_KEY` pre-loads every suit nibble with 3 (`0x3333 << 48`), the
///     flush-bias from section 2c.
///
/// Indices run 0..=51 (`suit*13 + value`); the 64-wide arrays leave 52..=63 zero
/// so a masked-to-6-bit index is always in bounds.
fn build_card_contributions() -> ([u64; 64], [u64; 64], u64) {
    let mut cards_key = [0u64; 64];
    let mut cards_mask = [0u64; 64];
    for suit in 0u32..4 {
        for value in 0u32..13 {
            let idx = (suit * 13 + value) as usize;
            cards_key[idx] = RANKS[value as usize] as u64 + (1u64 << (48 + 4 * suit));
            cards_mask[idx] = 1u64 << idx;
        }
    }
    let default_key: u64 = 0x3333u64 << 48;
    (cards_key, cards_mask, default_key)
}

// ============================================================================
// Self-checks: invariants the runtime depends on
//
// Each runs at build time and panics (failing the build) on violation, so a
// broken table can never ship.
// ============================================================================

/// Prove the `RANKS` multipliers are a perfect hash on histograms: every
/// reachable histogram has a distinct `rank_key`. See section 3.
fn assert_rank_keys_distinct(entries: &[ClassEntry]) {
    let mut keys: Vec<u32> = entries.iter().map(|e| e.0).collect();
    keys.sort_unstable();
    for w in keys.windows(2) {
        assert_ne!(
            w[0], w[1],
            "rank key collision: multipliers are not perfect"
        );
    }
}

/// Prove the perfect hash round-trips: every live key reads back its own score.
fn assert_hash_readback(value_of: &ValueMap, lookup: &[u16], offsets: &[u32]) {
    for (&k, &score) in value_of {
        let slot = perf_slot(k, offsets[(k >> ROW_SHIFT) as usize]);
        assert_eq!(lookup[slot], score, "perf-hash readback mismatch");
    }
}

/// Prove the tables collapse onto exactly the 7462 distinct five-card hands.
///
/// Enumerates every five-card hand both ways -- non-flush histograms (total 5)
/// resolved through `value_of`, and flush patterns (exactly 5 bits) resolved
/// through `flush_lookup` -- collects the distinct scores, and checks the count.
/// 7462 is the textbook number of five-card equivalence classes.
fn assert_7462_classes(value_of: &ValueMap, flush_lookup: &[u16]) {
    let mut classes: HashSet<u16> = HashSet::new();

    let mut c = [0u8; 13];
    enumerate_counts_exact(&mut c, 0, 5, &mut |counts| {
        classes.insert(value_of[&rank_key(counts)]);
    });
    for pattern in 0u32..(FLUSH_LEN as u32) {
        if pattern.count_ones() == 5 {
            classes.insert(flush_lookup[pattern as usize]);
        }
    }

    assert_eq!(classes.len(), 7462, "expected 7462 equivalence classes");
}

// ============================================================================
// Code emission
// ============================================================================

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Build every table.
    let (value_of, entries) = build_value_map();
    let (lookup, offsets) = build_perfect_hash(&value_of);
    let flush_lookup = build_flush_table();
    let (cards_key, cards_mask, default_key) = build_card_contributions();

    // Verify every invariant the runtime relies on before emitting anything.
    assert_rank_keys_distinct(&entries);
    assert_hash_readback(&value_of, &lookup, &offsets);
    assert_7462_classes(&value_of, &flush_lookup);

    // Emit the generated source.
    let mut out = String::new();
    out.push_str("// @generated by build.rs. Do not edit.\n");
    writeln!(out, "pub(crate) const LOOKUP_LEN: usize = {LOOKUP_LEN};").unwrap();
    writeln!(out, "pub(crate) const FLUSH_LEN: usize = {FLUSH_LEN};").unwrap();
    writeln!(out, "pub(crate) const ROW_SHIFT: u32 = {ROW_SHIFT};").unwrap();
    writeln!(
        out,
        "pub(crate) const ROW_OFFSETS_LEN: usize = {ROW_OFFSETS_LEN};"
    )
    .unwrap();
    writeln!(out, "pub(crate) const DEFAULT_KEY: u64 = {default_key};").unwrap();
    emit_u16_array(&mut out, "LOOKUP", &lookup);
    emit_u16_array(&mut out, "FLUSH_LOOKUP", &flush_lookup);
    emit_u32_array(&mut out, "ROW_OFFSETS", &offsets);
    emit_u64_array(&mut out, "CARDS_KEY", &cards_key);
    emit_u64_array(&mut out, "CARDS_MASK", &cards_mask);

    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("eval_tables.rs");
    fs::write(dest, out).unwrap();
}

/// Emit `pub(crate) static NAME: [u16; N] = [ ... ];`, 32 values per line.
fn emit_u16_array(out: &mut String, name: &str, data: &[u16]) {
    writeln!(out, "pub(crate) static {name}: [u16; {}] = [", data.len()).unwrap();
    for chunk in data.chunks(32) {
        for v in chunk {
            write!(out, "{v},").unwrap();
        }
        out.push('\n');
    }
    out.push_str("];\n");
}

/// Emit `pub(crate) static NAME: [u32; N] = [ ... ];`, 32 values per line.
fn emit_u32_array(out: &mut String, name: &str, data: &[u32]) {
    writeln!(out, "pub(crate) static {name}: [u32; {}] = [", data.len()).unwrap();
    for chunk in data.chunks(32) {
        for v in chunk {
            write!(out, "{v},").unwrap();
        }
        out.push('\n');
    }
    out.push_str("];\n");
}

/// Emit `pub(crate) static NAME: [u64; N] = [ ... ];`, 16 values per line.
fn emit_u64_array(out: &mut String, name: &str, data: &[u64]) {
    writeln!(out, "pub(crate) static {name}: [u64; {}] = [", data.len()).unwrap();
    for chunk in data.chunks(16) {
        for v in chunk {
            write!(out, "{v},").unwrap();
        }
        out.push('\n');
    }
    out.push_str("];\n");
}
