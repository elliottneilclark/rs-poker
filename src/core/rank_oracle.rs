// Frozen pre-perfect-hash evaluator, kept only to validate and benchmark the
// new one. Returns an old-style packed u32 (category 0..8 in bits 28.., bitmask
// payload below) so ordering and category can be cross-checked.
//
// This file is not a module of its own: it is `include!`d both by the
// `cfg(test)` `oracle` module in rank.rs and by benches/oracle_rank.rs, so the
// tested oracle and the benchmarked oracle cannot drift apart.

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
