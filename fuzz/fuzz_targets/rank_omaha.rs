#![no_main]
#[macro_use]
extern crate libfuzzer_sys;
extern crate rs_poker;
use rs_poker::core::{CardBitSet, CardIter, Deck, RankFive, Rankable};
use rs_poker::omaha::OmahaHand;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: one for hole count, one for card selection
    if data.len() < 2 {
        return;
    }

    // Determine hole card count (2-7) and board card count (3-5)
    let hole_count = (data[0] % 6) as usize + 2; // 2..=7
    let board_count = (data[1] % 3) as usize + 3; // 3..=5
    let total = hole_count + board_count;

    if data.len() < 2 + total {
        return;
    }

    // Use bytes to select cards from the deck without replacement
    let deck = Deck::default();
    let all_cards: Vec<_> = deck.into_iter().collect();
    let mut used = [false; 52];
    let mut selected = Vec::with_capacity(total);

    for &b in &data[2..2 + total] {
        let start = (b as usize) % 52;
        // Find next unused card from start position
        let mut found = false;
        for offset in 0..52 {
            let idx = (start + offset) % 52;
            if !used[idx] {
                used[idx] = true;
                selected.push(all_cards[idx]);
                found = true;
                break;
            }
        }
        if !found {
            return;
        }
    }

    let hole: CardBitSet = selected[..hole_count].iter().copied().collect();
    let board: CardBitSet = selected[hole_count..].iter().copied().collect();

    let hand = match OmahaHand::new(hole, board) {
        Ok(h) => h,
        Err(_) => return,
    };

    // Oracle: brute-force all C(h,2) * C(b,3) combos
    let oracle_rank = CardIter::new(hole, 2)
        .flat_map(|h| CardIter::new(board, 3).map(move |b| (h | b).rank_five()))
        .max()
        .unwrap();

    let actual_rank = hand.rank();
    assert_eq!(oracle_rank, actual_rank);
});
