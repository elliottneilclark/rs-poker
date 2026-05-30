use rand::Rng;

use crate::arena::{GameState, action::AgentAction, game_state::Round};
use crate::core::{Card, Deck, PlayerBitSet, Rank, Rankable, SevenCardAccum, Suit, Value};

/// Upper bound on simultaneous contenders. `PlayerBitSet` is `u16`-backed, so a
/// table seats at most 16 players; contenders never exceed that. Lets the
/// showdown enumeration tally bases on the stack instead of in a heap `Vec`.
const MAX_CONTENDERS: usize = 16;

/// Upper bound on cards left in the deck (a full deck is 52). Lets board
/// enumeration collect the remaining cards into a stack buffer.
const DECK_LEN: usize = 52;

// -----------------------------------------------------------------------------
// Fast-forward helpers
//
// These free functions implement the cheap reward path used by
// `CFRAgent::compute_reward_fast_forward`. They mutate a cloned `GameState`
// directly: apply the candidate action, play out the rest of the hand
// assuming every further action is a check/call, and distribute a single pot.
// -----------------------------------------------------------------------------

/// Apply a single action on behalf of the current to-act player.
///
/// If the action fails validation (e.g. an illegal raise size because the
/// game state has drifted), we fall back to calling the current bet. An
/// unreachable edge case should degrade gracefully rather than poisoning
/// the reward signal.
pub(super) fn fast_forward_apply_action(gs: &mut GameState, action: &AgentAction) {
    let call_current_bet = |gs: &mut GameState| {
        let _ = gs.do_bet(gs.current_round_bet(), false);
    };
    match action {
        AgentAction::Fold => gs.fold(),
        AgentAction::Call => call_current_bet(gs),
        AgentAction::Bet(amount) => {
            if gs.do_bet(*amount, false).is_err() {
                call_current_bet(gs);
            }
        }
        AgentAction::AllIn => {
            let idx = gs.to_act_idx();
            let target = gs.stacks[idx] + gs.current_round_player_bet(idx);
            if gs.do_bet(target, false).is_err() {
                call_current_bet(gs);
            }
        }
    }
}

/// Walk the game state forward through any remaining rounds. Betting rounds
/// are settled by having every still-needing-action player call; deal rounds
/// are settled by drawing fresh community cards from the remaining deck.
pub(super) fn fast_forward_run_to_showdown<R: Rng>(gs: &mut GameState, rng: &mut R) {
    let mut deck = fast_forward_remaining_deck(gs);
    loop {
        // If at most one player can contest the pot, further play is moot —
        // skip straight to the pot distribution step.
        let contenders = gs.player_active.count() + gs.player_all_in.count();
        if contenders <= 1 {
            return;
        }
        match gs.round {
            Round::Showdown | Round::Complete => return,
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            Round::DealFlop => {
                fast_forward_deal_community_cards(gs, &mut deck, 3, rng);
                gs.advance_round();
            }
            Round::DealTurn | Round::DealRiver => {
                fast_forward_deal_community_cards(gs, &mut deck, 1, rng);
                gs.advance_round();
            }
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Have every player whose `needs_action` bit is still set call the current
/// bet. Players who cannot cover the call still put in what they have and
/// are marked all-in by `do_bet`.
fn fast_forward_everyone_calls(gs: &mut GameState) {
    // Safety cap: at most one call per seat per round. `do_bet` disables the
    // to-act player's `needs_action` bit, so this loop must terminate in at
    // most `num_players` iterations.
    for _ in 0..gs.num_players {
        if gs.round_data.num_players_need_action() == 0 {
            break;
        }
        let to_match = gs.current_round_bet();
        if gs.do_bet(to_match, false).is_err() {
            // The call validator can reject in pathological states (e.g.
            // NaN). Fall back to a check so we don't loop forever.
            let _ = gs.do_bet(0.0, false);
        }
    }
}

/// Build a deck of cards that haven't been dealt yet by removing every known
/// card from a fresh 52-card deck. Each player's hand already contains the
/// shared board cards, so iterating hands covers the board implicitly.
fn fast_forward_remaining_deck(gs: &GameState) -> Deck {
    let mut deck = Deck::default();
    for hand in &gs.hands {
        for card in hand.iter() {
            deck.remove(&card);
        }
    }
    deck
}

/// Draw `num_cards` from the deck and add them to the board and to every
/// player's hand, mirroring what `HoldemSimulation::deal_comunity_cards` does.
fn fast_forward_deal_community_cards<R: Rng>(
    gs: &mut GameState,
    deck: &mut Deck,
    num_cards: usize,
    rng: &mut R,
) {
    for _ in 0..num_cards {
        let Some(card) = deck.deal(rng) else { return };
        gs.board.push(card);
        for hand in gs.hands.iter_mut() {
            hand.insert(card);
        }
    }
}

/// Award the full pot to the best hand(s) among players still in the pot.
/// Uses a single pot (no side pots): ties split evenly.
pub(super) fn fast_forward_distribute_pot(gs: &mut GameState) {
    let contenders = gs.player_active | gs.player_all_in;
    let count = contenders.count();
    if count == 0 {
        return;
    }
    let pot = gs.total_pot;
    if pot <= 0.0 {
        return;
    }
    if count == 1 {
        let winner = contenders.ones().next().unwrap();
        gs.award(winner, pot);
        gs.total_pot = 0.0;
        return;
    }
    let winners = find_winners(&contenders, &gs.hands);
    let split = pot / winners.count() as f32;
    for idx in winners.ones() {
        gs.award(idx, split);
    }
    gs.total_pot = 0.0;
}

/// Rank each contender's hand and return the set of player indices that share
/// the best rank. Ties are reported as multiple winners; the caller decides how
/// to split the pot.
fn find_winners(contenders: &PlayerBitSet, hands: &[crate::core::Hand]) -> PlayerBitSet {
    let mut best_rank = None;
    let mut winners = PlayerBitSet::default();
    for idx in contenders.ones() {
        let rank = hands[idx].rank();
        match best_rank {
            None => {
                best_rank = Some(rank);
                winners.enable(idx);
            }
            Some(current) if rank > current => {
                best_rank = Some(rank);
                winners = PlayerBitSet::default();
                winners.enable(idx);
            }
            Some(current) if rank == current => winners.enable(idx),
            _ => {}
        }
    }
    winners
}

/// Advance the game state through all remaining betting rounds (everyone
/// calls/checks) until a deal round or showdown is reached. This separates
/// the deterministic betting from the stochastic card dealing, allowing
/// the caller to enumerate board completions instead of sampling.
pub(super) fn fast_forward_advance_betting(gs: &mut GameState) {
    // Safety cap: at most 8 round advances to prevent infinite loops.
    for _ in 0..8 {
        match gs.round {
            // Stop at deal rounds — the caller will enumerate cards.
            Round::DealFlop | Round::DealTurn | Round::DealRiver => return,
            // Stop at terminal states.
            Round::Showdown | Round::Complete => return,
            // Skip non-betting advance rounds.
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            // Betting rounds: everyone calls, then advance.
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Reward for `player_idx` when at most one player can still contest the pot.
///
/// Called after `fast_forward_advance_betting` has moved bets into the pot.
/// Returns `None` when two or more players remain, signalling that a real
/// showdown (enumeration or sampling) is still required. The single-contender
/// branch includes `gs.stacks[player_idx]` because chips have already moved
/// from stacks into the pot.
fn fast_forward_uncontested_reward(
    gs: &GameState,
    contenders: PlayerBitSet,
    player_idx: usize,
) -> Option<f32> {
    match contenders.count() {
        0 => Some(gs.player_reward(player_idx)),
        1 => {
            let winner = contenders.ones().next().unwrap();
            let winnings = if winner == player_idx {
                gs.total_pot
            } else {
                0.0
            };
            Some(gs.stacks[player_idx] + winnings - gs.starting_stacks[player_idx])
        }
        _ => None,
    }
}

/// Enumerate all possible board completions and compute the exact expected
/// reward for `player_idx`.
///
/// This replaces the random-sample approach in `fast_forward_run_to_showdown`
/// with deterministic enumeration when the number of remaining cards is small
/// enough (0, 1, or 2 cards). The result is zero variance in the reward
/// signal, which dramatically improves CFR convergence quality.
///
/// # Arguments
///
/// * `gs` - Game state positioned at a deal round (or showdown) after all
///   betting is resolved. The `total_pot` must already reflect all bets.
/// * `player_idx` - The player whose reward we compute.
/// * `cards_needed` - Number of community cards still to be dealt (0, 1, or 2).
pub(super) fn fast_forward_enumerate_showdowns(
    gs: &GameState,
    player_idx: usize,
    cards_needed: usize,
) -> f32 {
    let contenders = gs.player_active | gs.player_all_in;

    // No contenders (everyone folded; pot already awarded) or a single
    // contender (wins regardless of board) needs no enumeration.
    if let Some(reward) = fast_forward_uncontested_reward(gs, contenders, player_idx) {
        return reward;
    }

    let pot = gs.total_pot;
    if pot <= 0.0 {
        return gs.player_reward(player_idx);
    }

    if cards_needed == 0 {
        // Board is complete — just evaluate the showdown.
        return evaluate_showdown_reward(gs, &contenders, pot, player_idx);
    }

    // Collect the remaining deck into a stack buffer for indexed access.
    let deck = fast_forward_remaining_deck(gs);
    let mut card_buf = [Card::new(Value::Two, Suit::Spade); DECK_LEN];
    let mut rn = 0;
    for c in deck.iter() {
        card_buf[rn] = c;
        rn += 1;
    }
    let remaining = &card_buf[..rn];

    let starting_stack = gs.starting_stacks[player_idx];
    // After fast_forward_advance_betting, chips have moved from stacks into
    // the pot. `evaluate_with_extra_cards` returns only the player's share of
    // the pot (or 0), so the net reward is:
    //   remaining_stack + pot_share - starting_stack
    // The remaining_stack term accounts for the chips the player kept — without
    // it the reward would be off by exactly the unbet portion of their stack.
    let remaining_stack = gs.stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut count = 0u32;

    // Tally each contender's fixed hole+board cards once; every runout below
    // only folds in the 1-2 enumerated cards on a cheap copy of the tally.
    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let base = contender_accums(gs, &contenders, &mut acc_buf);
    if cards_needed == 1 {
        // Enumerate single card (river).
        for &card in remaining {
            let reward = combo_reward::<1>(base, player_idx, pot, [card]);
            total_reward += f64::from(remaining_stack + reward - starting_stack);
            count += 1;
        }
    } else {
        // cards_needed == 2: enumerate all unordered pairs (turn + river).
        // Card order doesn't matter for hand evaluation, so visit each once.
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward = combo_reward::<2>(base, player_idx, pot, [remaining[i], remaining[j]]);
                total_reward += f64::from(remaining_stack + reward - starting_stack);
                count += 1;
            }
        }
    }

    (total_reward / f64::from(count)) as f32
}

/// Number of random flop samples to draw when 3 community cards remain.
/// For each sampled flop, all turn+river combinations are enumerated
/// exhaustively (~C(44,2) ≈ 946 evals per flop). This hybrid approach
/// gives much lower variance than a single random runout at modest cost.
pub(super) const FLOP_SAMPLES: usize = 3;

/// Sample random flops and enumerate all turn+river completions for each.
///
/// When 3 community cards remain (pre-flop fast-forward), full enumeration
/// costs C(47,3) ≈ 16K evaluations — too expensive per action. Instead we
/// sample `FLOP_SAMPLES` random flop combinations and for each one
/// exhaustively enumerate all C(remaining,2) turn+river pairs. This
/// eliminates variance from 2 of the 3 unknown cards while keeping cost
/// at roughly `FLOP_SAMPLES × 1000` evaluations.
pub(super) fn fast_forward_sample_flop_enumerate_runout<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
) -> f32 {
    fast_forward_sample_flop_enumerate_runout_n(gs, player_idx, rng, FLOP_SAMPLES)
}

/// Inner implementation parameterized by sample count for benchmarking.
pub(super) fn fast_forward_sample_flop_enumerate_runout_n<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
    num_samples: usize,
) -> f32 {
    let contenders = gs.player_active | gs.player_all_in;

    // No contenders (everyone folded) or a single contender (wins regardless of
    // board) needs no sampling.
    if let Some(reward) = fast_forward_uncontested_reward(gs, contenders, player_idx) {
        return reward;
    }

    let pot = gs.total_pot;
    if pot <= 0.0 {
        return gs.player_reward(player_idx);
    }

    let mut deck = fast_forward_remaining_deck(gs);
    let starting_stack = gs.starting_stacks[player_idx];
    // See comment in fast_forward_enumerate_showdowns — remaining_stack
    // accounts for unbet chips after fast_forward_advance_betting.
    let remaining_stack = gs.stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut total_count = 0u64;

    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let mut card_buf = [Card::new(Value::Two, Suit::Spade); DECK_LEN];
    for _ in 0..num_samples {
        // Deal 3 random flop cards from the deck.
        let mut flop = [Card::new(Value::Two, Suit::Spade); 3];
        let mut fc = 0;
        while fc < 3 {
            match deck.deal(rng) {
                Some(c) => {
                    flop[fc] = c;
                    fc += 1;
                }
                // Not enough cards — shouldn't happen in practice.
                None => break,
            }
        }
        if fc < 3 {
            break;
        }

        // Tally each contender's hole+flop cards once for this sampled flop.
        // No GameState clone is needed: `combo_reward` only reads the per-hand
        // tally, so we fold the hole cards and the flop straight into the
        // accumulator and skip the (allocation-heavy) board/hand mutation.
        let mut n = 0;
        for idx in contenders.ones() {
            let mut acc = SevenCardAccum::new();
            for c in gs.hands[idx].iter() {
                acc.add(c);
            }
            for &c in &flop {
                acc.add(c);
            }
            acc_buf[n] = (idx, acc);
            n += 1;
        }
        let base = &acc_buf[..n];

        // The flop cards were just drawn from `deck`, so it already holds
        // exactly the cards available for the turn/river. Collect them into a
        // stack buffer and enumerate all turn+river completions on top.
        let mut rn = 0;
        for c in deck.iter() {
            card_buf[rn] = c;
            rn += 1;
        }
        let remaining = &card_buf[..rn];
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward = combo_reward::<2>(base, player_idx, pot, [remaining[i], remaining[j]]);
                total_reward += f64::from(remaining_stack + reward - starting_stack);
                total_count += 1;
            }
        }

        // Put flop cards back in the deck for the next sample.
        for &card in &flop {
            deck.insert(card);
        }
    }

    if total_count == 0 {
        return gs.player_reward(player_idx);
    }

    (total_reward / total_count as f64) as f32
}

/// Precompute per-contender ranking state for the fixed hole+board cards.
///
/// Each contender's `gs.hands[idx]` already holds its hole cards plus every
/// community card dealt so far. Those cards are constant across a board
/// enumeration, so we tally them once into a [`SevenCardAccum`]; each
/// enumerated runout then only folds its 1-2 extra cards into a cheap `Copy`
/// of the tally rather than re-iterating the whole hand.
fn contender_accums<'b>(
    gs: &GameState,
    contenders: &PlayerBitSet,
    buf: &'b mut [(usize, SevenCardAccum); MAX_CONTENDERS],
) -> &'b [(usize, SevenCardAccum)] {
    let mut n = 0;
    for idx in contenders.ones() {
        let mut acc = SevenCardAccum::new();
        for c in gs.hands[idx].iter() {
            acc.add(c);
        }
        buf[n] = (idx, acc);
        n += 1;
    }
    &buf[..n]
}

/// Reward for `player_idx` on one board completion, given precomputed base
/// accumulators. Folds `extra` into each contender's tally, finds the best
/// rank, and returns `player_idx`'s share of `pot` (0 if not a winner).
///
/// `N` is the number of extra cards to fold in (0, 1, or 2). Making it a
/// const generic monomorphizes the inner `for` over a fixed-length array, so
/// the 1- and 2-card runout loops unroll the `add` calls and skip the slice
/// bounds checks the old `&[Card]` signature paid on every contender.
///
/// Equivalent to [`find_winners`] over hands extended with `extra` followed by
/// an even split of `pot`, but it allocates nothing and re-ranks only the
/// varying cards.
#[inline]
fn combo_reward<const N: usize>(
    base: &[(usize, SevenCardAccum)],
    player_idx: usize,
    pot: f32,
    extra: [crate::core::Card; N],
) -> f32 {
    let mut best: Option<Rank> = None;
    let mut win_count = 0u32;
    let mut player_wins = false;
    for &(idx, base_acc) in base {
        let mut acc = base_acc;
        for card in extra {
            acc.add(card);
        }
        let rank = acc.rank();
        match best {
            // A strictly worse hand never affects the winner set.
            Some(b) if rank < b => {}
            // A tie for the lead adds another contender to the split.
            Some(b) if rank == b => {
                win_count += 1;
                player_wins |= idx == player_idx;
            }
            // `None` (first contender) or a new strict leader resets the set.
            _ => {
                best = Some(rank);
                win_count = 1;
                player_wins = idx == player_idx;
            }
        }
    }
    if player_wins {
        pot / win_count as f32
    } else {
        0.0
    }
}

/// Evaluate showdown with the current board (no extra cards).
/// Returns `remaining_stack + pot_share - starting_stack` to account for
/// chips already moved from stacks into the pot by `fast_forward_advance_betting`.
fn evaluate_showdown_reward(
    gs: &GameState,
    contenders: &PlayerBitSet,
    pot: f32,
    player_idx: usize,
) -> f32 {
    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let base = contender_accums(gs, contenders, &mut acc_buf);
    let reward = combo_reward::<0>(base, player_idx, pot, []);
    gs.stacks[player_idx] + reward - gs.starting_stacks[player_idx]
}
