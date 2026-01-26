use std::collections::{HashMap, HashSet, VecDeque};

use approx::relative_eq;

use crate::core::Card;

use super::{Action, ActionObj, HandHistory, PlayerObj, PotObj, RoundObj};

/// Assert that an Open Hand History object is internally consistent.
///
/// This performs structural validation of betting flow, chip accounting,
/// card uniqueness, forced blinds correspondence, and general data-model sanity.
pub fn assert_valid_open_hand_history(hand_history: &HandHistory) {
    let mut validator = HandHistoryValidator::new(hand_history);
    validator.validate();
}

/// Assert that the Open Hand History mirrors the final arena `GameState`.
///
/// This runs the base validation and additionally checks chips, board cards,
/// player counts, and winnings against the provided `GameState`.
#[cfg(feature = "arena")]
pub fn assert_open_hand_history_matches_game_state(
    hand_history: &HandHistory,
    game_state: &crate::arena::GameState,
) {
    assert_valid_open_hand_history(hand_history);
    HandHistoryArenaComparator::new(hand_history, game_state).assert_consistent();
}

struct HandHistoryValidator<'a> {
    hh: &'a HandHistory,
    players: HashMap<u64, PlayerState>,
    active_order: Vec<u64>,
    dealer_player_id: u64,
    small_blind_player: Option<u64>,
    big_blind_player: Option<u64>,
    board_cards: Vec<Card>,
    seen_cards: HashSet<Card>,
    total_contribution: f32,
    table_contribution: f32,
    ante_posted: HashSet<u64>,
    small_blind_posted: bool,
    big_blind_posted: bool,
    betting_state: BettingRoundState,
    rotation: BettingRotation,
}

impl<'a> HandHistoryValidator<'a> {
    fn new(hh: &'a HandHistory) -> Self {
        assert!(
            !hh.players.is_empty(),
            "Hand {game} must include at least one player",
            game = hh.game_number
        );
        assert_eq!(
            hh.players.len() as u64,
            hh.table_size,
            "Hand {game} has mismatched table size",
            game = hh.game_number
        );
        assert!(
            hh.big_blind_amount + f32::EPSILON >= hh.small_blind_amount,
            "Hand {game} big blind must be >= small blind",
            game = hh.game_number
        );

        let mut players: HashMap<u64, PlayerState> = HashMap::new();
        let mut seat_entries = Vec::with_capacity(hh.players.len());
        for player in &hh.players {
            assert!(
                players
                    .insert(player.id, PlayerState::new(player))
                    .is_none(),
                "Hand {game} contains duplicate player id {id}",
                game = hh.game_number,
                id = player.id
            );
            seat_entries.push((player.seat, player.id));
        }
        seat_entries.sort_by_key(|(seat, _)| *seat);

        let dealer_player_id = seat_entries
            .iter()
            .find(|(seat, _)| *seat == hh.dealer_seat)
            .map(|(_, id)| *id)
            .expect("Dealer seat must correspond to a player");

        let active_order: Vec<u64> = seat_entries
            .iter()
            .map(|(_, id)| *id)
            .filter(|id| !players.get(id).unwrap().sitting_out)
            .collect();
        assert!(
            active_order.len() >= 2,
            "Hand {game} must have at least two active players",
            game = hh.game_number
        );
        assert!(
            active_order.contains(&dealer_player_id),
            "Dealer must be seated at the table"
        );

        let small_blind_player = determine_small_blind(&active_order, dealer_player_id);
        let big_blind_player =
            determine_big_blind(&active_order, dealer_player_id, small_blind_player);

        let rotation = BettingRotation::new(active_order.clone());

        Self {
            hh,
            players,
            active_order,
            dealer_player_id,
            small_blind_player,
            big_blind_player,
            board_cards: Vec::new(),
            seen_cards: HashSet::new(),
            total_contribution: 0.0,
            table_contribution: 0.0,
            ante_posted: HashSet::new(),
            small_blind_posted: false,
            big_blind_posted: false,
            betting_state: BettingRoundState::default(),
            rotation,
        }
    }

    fn validate(&mut self) {
        assert!(
            !self.hh.rounds.is_empty(),
            "Hand {game} must contain at least one round",
            game = self.hh.game_number
        );

        for round in &self.hh.rounds {
            self.process_round(round);
        }

        self.finish_validation();
    }

    fn process_round(&mut self, round: &RoundObj) {
        let street = Street::from(round.street.as_str());
        if street.is_betting_round() {
            self.start_betting_round(street);
        }

        if let Some(cards) = &round.cards {
            self.record_board_cards(street, cards);
        }

        for action in &round.actions {
            self.process_action(street, action);
        }
    }

    fn start_betting_round(&mut self, street: Street) {
        // In No-Limit Texas Hold'em, the minimum raise starts at the big blind
        self.betting_state.reset(self.hh.big_blind_amount);
        self.rotation.start_round(
            street,
            self.dealer_player_id,
            self.big_blind_player,
            &self.players,
        );
    }

    fn record_board_cards(&mut self, street: Street, cards: &[Card]) {
        match street {
            Street::Flop => assert_eq!(cards.len(), 3, "Flop must contain exactly 3 cards"),
            Street::Turn | Street::River => {
                assert_eq!(cards.len(), 1, "Turn and river must contain exactly 1 card")
            }
            _ => {
                assert!(cards.is_empty(), "Only community streets may specify cards");
                return;
            }
        }

        for &card in cards {
            self.assert_new_card(card, "board");
            self.board_cards.push(card);
        }

        assert!(
            self.board_cards.len() <= 5,
            "A holdem board cannot contain more than 5 cards"
        );
    }

    fn process_action(&mut self, street: Street, action: &ActionObj) {
        let player_id = action.player_id;
        self.players
            .get(&player_id)
            .unwrap_or_else(|| panic!("Unknown player {player_id} referenced"));

        assert!(action.amount.is_finite(), "Action amounts must be finite");
        assert!(
            action.amount >= 0.0 || matches!(action.action, Action::Fold | Action::Check),
            "Negative chip movements are invalid"
        );

        let requires_turn = matches!(
            action.action,
            Action::Fold | Action::Check | Action::Call | Action::Bet | Action::Raise
        ) && street.is_betting_round();
        if requires_turn {
            self.rotation.expect_actor(player_id, &self.players);
        }

        match action.action {
            Action::DealtCards => self.handle_dealt_cards(player_id, action),
            Action::PostAnte => self.handle_post_ante(player_id, action.amount),
            Action::PostSmallBlind => self.handle_small_blind(player_id, action.amount),
            Action::PostBigBlind => self.handle_big_blind(player_id, action.amount),
            Action::PostDead | Action::PostExtraBlind | Action::Straddle => {
                self.handle_optional_force(player_id, action.amount)
            }
            Action::Bet => self.handle_bet(player_id, action.amount, street, action.is_allin),
            Action::Raise => self.handle_raise(player_id, action.amount, action.is_allin),
            Action::Call => self.handle_call(player_id, action.amount, action.is_allin),
            Action::Check => self.handle_check(player_id, action.is_allin),
            Action::Fold => self.handle_fold(player_id),
            Action::AddedChips => self.handle_added_chips(player_id, action.amount),
            Action::AddedToPot => self.handle_table_addition(action.amount),
            Action::ShowsCards => self.handle_show_cards(player_id, action),
            Action::MucksCards => self.handle_muck(player_id),
            Action::SitsDown | Action::StandsUp => {}
        }

        if requires_turn {
            self.rotation.trim_inactive(&self.players);
        }
    }

    fn handle_dealt_cards(&mut self, player_id: u64, action: &ActionObj) {
        let cards = action
            .cards
            .as_ref()
            .expect("DealtCards entries must include card payloads");
        assert_eq!(
            cards.len(),
            2,
            "Holdem players must receive exactly 2 cards"
        );
        {
            let state = self
                .players
                .get(&player_id)
                .expect("Player must exist for DealtCards action");
            assert!(
                state.cards.is_empty(),
                "Player {player_id} already has hole cards assigned"
            );
        }

        for &card in cards {
            self.assert_new_card(card, "player");
        }

        let state = self
            .players
            .get_mut(&player_id)
            .expect("Player must exist for DealtCards action");
        state.cards.extend(cards.iter().copied());
    }

    fn handle_post_ante(&mut self, player_id: u64, amount: f32) {
        assert!(amount >= 0.0, "Ante amount must be non-negative");
        if self.hh.ante_amount > 0.0 {
            let stack_remaining = self.players.get(&player_id).unwrap().stack_remaining;
            assert!(
                approx_eq(amount, self.hh.ante_amount)
                    || stack_remaining + f32::EPSILON <= self.hh.ante_amount,
                "Player {player_id} posted incorrect ante (amount {amount}, expected {expected}, stack {stack_remaining})",
                expected = self.hh.ante_amount
            );
        }
        self.ante_posted.insert(player_id);
        self.apply_contribution(player_id, amount, "ante");
    }

    fn handle_small_blind(&mut self, player_id: u64, amount: f32) {
        if let Some(expected) = self.small_blind_player {
            assert_eq!(
                player_id, expected,
                "Small blind must be posted by expected player"
            );
        }
        self.small_blind_posted = true;
        self.validate_forced_amount(player_id, amount, self.hh.small_blind_amount);
        self.apply_contribution(player_id, amount, "small blind");
    }

    fn handle_big_blind(&mut self, player_id: u64, amount: f32) {
        if let Some(expected) = self.big_blind_player {
            assert_eq!(
                player_id, expected,
                "Big blind must be posted by expected player"
            );
        }
        self.big_blind_posted = true;
        self.validate_forced_amount(player_id, amount, self.hh.big_blind_amount);
        self.apply_contribution(player_id, amount, "big blind");
    }

    fn handle_optional_force(&mut self, player_id: u64, amount: f32) {
        assert!(amount >= 0.0, "Forced bets must be non-negative");
        self.apply_contribution(player_id, amount, "forced bet");
    }

    fn ensure_effective_wager_amount(
        &self,
        player_id: u64,
        amount: f32,
        is_allin: bool,
        label: &str,
    ) {
        if amount > f32::EPSILON {
            return;
        }

        assert!(
            amount > 0.0,
            "{label} amount must be positive",
            label = label
        );
        let state = self.players.get(&player_id).unwrap();
        let short_stack_threshold = f32::EPSILON + f32::EPSILON;
        assert!(
            is_allin && state.stack_remaining <= short_stack_threshold,
            "Player {player_id} {label} amount {amount} below minimum ({min_amount}) without being all-in (stack {stack})",
            player_id = player_id,
            label = label,
            amount = amount,
            min_amount = f32::EPSILON,
            stack = state.stack_remaining,
        );
    }

    fn handle_bet(&mut self, player_id: u64, amount: f32, street: Street, is_allin: bool) {
        assert!(
            street.is_betting_round(),
            "Bets only allowed on betting streets"
        );
        self.ensure_effective_wager_amount(player_id, amount, is_allin, "bet");
        assert!(
            self.betting_state.current_max <= f32::EPSILON,
            "Cannot bet when a live bet exists"
        );
        self.ensure_player_can_act(player_id, "bet");

        // Validate minimum bet sizing (Texas Hold'em No-Limit rule):
        // An opening bet must be at least the big blind, unless it's an all-in
        if !is_allin && self.hh.big_blind_amount > 0.0 {
            let min_bet = self.hh.big_blind_amount;
            // Allow small tolerance for floating point arithmetic
            let tolerance = min_bet * 0.001 + f32::EPSILON;
            assert!(
                amount >= min_bet - tolerance,
                "Player {} bet of {} does not meet minimum bet requirement of {}",
                player_id,
                amount,
                min_bet
            );
        }

        self.apply_contribution(player_id, amount, "bet");
        self.rotation.rebuild_after_raise(player_id, &self.players);
    }

    fn handle_raise(&mut self, player_id: u64, amount: f32, is_allin: bool) {
        self.ensure_effective_wager_amount(player_id, amount, is_allin, "raise");
        assert!(
            self.betting_state.current_max > 0.0,
            "Cannot raise without a live bet"
        );
        self.ensure_player_can_act(player_id, "raise");
        let previous_max = self.betting_state.current_max;
        let already_committed = self.betting_state.committed(player_id);
        let new_total = self.apply_contribution(player_id, amount, "raise");

        // The raise amount is how much this raises the current bet
        let raise_amount = new_total - previous_max;

        // For a raise to be valid:
        // - Normal raise: must exceed current bet by at least f32::EPSILON
        // - All-in raise: just needs to exceed current bet (with floating point tolerance
        //   in the player's favor to handle cases where remaining chips are tiny)
        let raise_is_valid = new_total > previous_max + f32::EPSILON
            || (is_allin && new_total + f32::EPSILON > previous_max);
        assert!(raise_is_valid, "Raise must exceed the current bet");

        // Validate minimum raise sizing (Texas Hold'em No-Limit rule):
        // A raise must be at least the size of the previous raise (or big blind for first raise)
        // All-in raises are exempt from minimum raise requirements
        if !is_allin && raise_amount > 0.0 {
            let min_raise = self.betting_state.min_raise;
            // Allow small tolerance for floating point arithmetic
            let tolerance = min_raise * 0.001 + f32::EPSILON;
            assert!(
                raise_amount >= min_raise - tolerance,
                "Player {} raise of {} does not meet minimum raise requirement of {} (committed: {}, previous max: {}, new total: {})",
                player_id,
                raise_amount,
                min_raise,
                already_committed,
                previous_max,
                new_total
            );
        }

        self.rotation.rebuild_after_raise(player_id, &self.players);
    }

    fn handle_call(&mut self, player_id: u64, amount: f32, is_allin: bool) {
        assert!(amount > 0.0, "Call amount must be positive");
        let current_max = self.betting_state.current_max;
        self.ensure_player_can_act(player_id, "call");
        let available = self
            .players
            .get(&player_id)
            .map(|state| state.stack_remaining)
            .unwrap_or(0.0);
        let committing_stack = is_allin || amount + f32::EPSILON >= available;
        let already = self.betting_state.committed(player_id);
        let required = (current_max - already).max(0.0);
        let has_live_bet =
            current_max > f32::EPSILON || required > 0.0 || (committing_stack && current_max > 0.0);
        assert!(has_live_bet, "Cannot call when no bet is pending");
        let catastrophic_slop = current_max.abs() * f32::EPSILON;
        assert!(
            approx_eq(required, amount)
                || amount >= required - (f32::EPSILON + catastrophic_slop)
                || committing_stack,
            "Player {player_id} attempted to call incorrect amount"
        );
        let new_total = self.apply_contribution(player_id, amount, "call");
        if !committing_stack {
            assert!(
                approx_eq(new_total, current_max) || new_total >= current_max - f32::EPSILON,
                "Call did not match outstanding bet"
            );
        }
    }

    fn handle_check(&mut self, player_id: u64, is_allin: bool) {
        let committed = self.betting_state.committed(player_id);
        assert!(
            approx_eq(committed, self.betting_state.current_max),
            "Player {player_id} checked while facing a bet"
        );
        // If player is marked as all-in on a check, they've depleted their stack.
        // Mark them all-in and remove from rotation so they don't act on future streets.
        if is_allin {
            if let Some(state) = self.players.get_mut(&player_id) {
                state.all_in = true;
                state.stack_remaining = 0.0;
            }
            self.rotation.remove_player(player_id);
        }
    }

    fn handle_fold(&mut self, player_id: u64) {
        let state = self
            .players
            .get_mut(&player_id)
            .expect("Fold action player must exist");
        assert!(!state.folded, "Player {player_id} cannot fold twice");
        assert!(!state.all_in, "All-in players cannot fold");
        state.folded = true;
        self.rotation.remove_player(player_id);
    }

    fn handle_added_chips(&mut self, player_id: u64, amount: f32) {
        assert!(amount >= 0.0, "Added chips must be non-negative");
        let state = self
            .players
            .get_mut(&player_id)
            .expect("Added chips player must exist");
        state.stack_remaining += amount;
        state.total_added_chips += amount;
        if amount > 0.0 {
            state.all_in = false;
        }
    }

    fn handle_table_addition(&mut self, amount: f32) {
        assert!(amount >= 0.0, "Added pot chips must be non-negative");
        self.table_contribution += amount;
    }

    fn handle_show_cards(&self, player_id: u64, action: &ActionObj) {
        if let Some(cards) = &action.cards {
            let state = self
                .players
                .get(&player_id)
                .expect("Show cards requires valid player");
            if !state.cards.is_empty() {
                assert_eq!(
                    state.cards.len(),
                    cards.len(),
                    "Player {player_id} revealed mismatched card count"
                );
                for card in cards {
                    assert!(
                        state.cards.contains(card),
                        "Player {player_id} revealed unexpected card"
                    );
                }
            }
        }
    }

    fn handle_muck(&self, _player_id: u64) {}

    fn finish_validation(&self) {
        if self.hh.ante_amount > 0.0 {
            for player_id in &self.active_order {
                assert!(
                    self.ante_posted.contains(player_id),
                    "Active player {player_id} failed to post ante"
                );
            }
        }

        if self.hh.small_blind_amount > 0.0 {
            assert!(self.small_blind_posted, "Small blind was not posted");
        }
        if self.hh.big_blind_amount > 0.0 {
            assert!(self.big_blind_posted, "Big blind was not posted");
        }

        // Validate Texas Hold'em specific rules
        self.validate_board_progression();
        self.validate_dealer_positioning();
        self.validate_betting_round_sequence();
        self.validate_raise_sizing();
        self.validate_player_hole_cards();

        let mut payouts: HashMap<u64, f32> = HashMap::new();
        let mut pot_total = 0.0;
        let mut total_rake = 0.0;
        let mut total_jackpot = 0.0;
        for pot in &self.hh.pots {
            pot_total += pot.amount;
            total_rake += pot.rake.unwrap_or(0.0);
            total_jackpot += pot.jackpot.unwrap_or(0.0);
            self.validate_pot(pot, &mut payouts);
        }

        let all_contributions = self.total_contribution + self.table_contribution;
        assert!(
            approx_eq(all_contributions, pot_total),
            "Total contributions {all_contrib} do not equal pot total {pot_total}",
            all_contrib = all_contributions
        );

        let payout_sum: f32 = payouts.values().copied().sum();
        let expected_payout = pot_total - total_rake - total_jackpot;
        assert!(
            approx_eq(payout_sum, expected_payout),
            "Winnings {payout_sum} must equal pot total minus rake and jackpot {expected}",
            expected = expected_payout
        );

        for (player_id, state) in &self.players {
            assert!(
                state.stack_remaining + f32::EPSILON >= 0.0,
                "Player {player_id} ended with negative chips"
            );
        }
    }

    fn validate_pot(&self, pot: &PotObj, payouts: &mut HashMap<u64, f32>) {
        let mut sum = 0.0;
        for win in &pot.player_wins {
            assert!(win.win_amount >= 0.0, "Pot wins must be non-negative");
            let player = self
                .players
                .get(&win.player_id)
                .expect("Pot winner must be a known player");
            assert!(
                !player.folded,
                "Folded player {} cannot win a pot",
                win.player_id
            );
            *payouts.entry(win.player_id).or_default() += win.win_amount;
            sum += win.win_amount;
        }

        let rake = pot.rake.unwrap_or(0.0);
        let jackpot = pot.jackpot.unwrap_or(0.0);
        assert!(
            approx_eq(sum + rake + jackpot, pot.amount),
            "Pot {} does not balance",
            pot.number
        );
    }

    fn validate_forced_amount(&self, player_id: u64, amount: f32, expected: f32) {
        if expected <= 0.0 {
            return;
        }
        let state = self.players.get(&player_id).unwrap();
        if state.starting_stack + state.total_added_chips + f32::EPSILON >= expected {
            assert!(
                approx_eq(amount, expected) || amount >= expected - f32::EPSILON,
                "Player {player_id} forced bet should match expected amount"
            );
        }
    }

    fn ensure_player_can_act(&self, player_id: u64, label: &str) {
        let state = self.players.get(&player_id).unwrap();
        assert!(
            !state.sitting_out,
            "Sitting out player {player_id} cannot {label}"
        );
        assert!(
            !state.folded,
            "Player {player_id} cannot {label} after folding"
        );
        assert!(!state.all_in, "All-in player {player_id} cannot {label}");
    }

    fn apply_contribution(&mut self, player_id: u64, amount: f32, label: &str) -> f32 {
        assert!(amount >= 0.0, "{label} amount must be non-negative");
        let state = self
            .players
            .get_mut(&player_id)
            .expect("Contribution player must exist");
        assert!(
            state.stack_remaining + f32::EPSILON >= amount,
            "Player {player_id} attempted to {label} more chips than available (amount {amount}, stack {stack}, contributed {contrib}, starting {starting})",
            amount = amount,
            stack = state.stack_remaining,
            contrib = state.total_contribution,
            starting = state.starting_stack
        );
        state.stack_remaining -= amount;
        state.total_contribution += amount;
        if state.stack_remaining <= f32::EPSILON {
            state.stack_remaining = 0.0;
            state.all_in = true;
        }
        self.total_contribution += amount;
        self.betting_state.record(player_id, amount)
    }

    fn assert_new_card(&mut self, card: Card, location: &str) {
        assert!(
            self.seen_cards.insert(card),
            "Duplicate card {:?} observed on {}",
            card,
            location
        );
    }

    /// Validate that board cards follow Texas Hold'em progression (0->3->4->5)
    fn validate_board_progression(&self) {
        let board_count = self.board_cards.len();
        assert!(
            matches!(board_count, 0 | 3 | 4 | 5),
            "Board must have 0, 3, 4, or 5 cards in Texas Hold'em, found {}",
            board_count
        );
    }

    /// Validate dealer positioning follows clockwise rotation rules
    fn validate_dealer_positioning(&self) {
        // Dealer must be at a valid position
        assert!(
            self.active_order.contains(&self.dealer_player_id),
            "Dealer player {} must be active in the hand",
            self.dealer_player_id
        );

        // For heads-up, dealer is small blind
        if self.active_order.len() == 2 {
            assert_eq!(
                self.small_blind_player,
                Some(self.dealer_player_id),
                "In heads-up play, dealer must be small blind"
            );
        }
    }

    /// Validate betting rounds follow proper sequence (preflop -> flop -> turn -> river -> showdown)
    fn validate_betting_round_sequence(&self) {
        let mut seen_streets = HashSet::new();
        let mut last_street = None;

        for round in &self.hh.rounds {
            let street = Street::from(round.street.as_str());
            seen_streets.insert(street);

            if let Some(prev_street) = last_street {
                assert!(
                    street.comes_after(prev_street) || street == prev_street,
                    "Street sequence violation: {:?} cannot follow {:?}",
                    street,
                    prev_street
                );
            }
            last_street = Some(street);
        }

        // If we have community cards, we must have seen the appropriate streets
        if matches!(self.board_cards.len(), 3..=5) {
            assert!(
                seen_streets.contains(&Street::Flop),
                "Flop street required for community cards"
            );
        }
    }

    /// Validate minimum raise sizing and blind structure
    fn validate_raise_sizing(&self) {
        if self.hh.big_blind_amount > 0.0 && self.hh.small_blind_amount > 0.0 {
            assert!(
                self.hh.big_blind_amount >= self.hh.small_blind_amount,
                "Big blind {} must be at least as large as small blind {}",
                self.hh.big_blind_amount,
                self.hh.small_blind_amount
            );
        }
    }

    /// Validate player hole cards (each active player should have exactly 2 cards)
    fn validate_player_hole_cards(&self) {
        for (player_id, state) in &self.players {
            if !state.sitting_out {
                // In a completed hand, each player should have been dealt exactly 2 cards
                // unless they were sitting out from the beginning
                assert!(
                    state.cards.len() <= 2,
                    "Player {} has {} hole cards, maximum is 2",
                    player_id,
                    state.cards.len()
                );

                // If the hand proceeded past dealing, all active players should have cards
                if !self.hh.rounds.is_empty() && state.cards.is_empty() && !state.folded {
                    // This is suspicious - player was active but never got cards
                    eprintln!("Warning: Active player {} has no hole cards", player_id);
                }
            }
        }
    }
}

fn determine_small_blind(active_order: &[u64], dealer_id: u64) -> Option<u64> {
    if active_order.len() < 2 {
        return None;
    }
    if active_order.len() == 2 {
        return Some(dealer_id);
    }
    next_after(active_order, dealer_id)
}

fn determine_big_blind(
    active_order: &[u64],
    dealer_id: u64,
    small_blind: Option<u64>,
) -> Option<u64> {
    if active_order.len() < 2 {
        return None;
    }
    if active_order.len() == 2 {
        return active_order.iter().copied().find(|id| *id != dealer_id);
    }
    let sb = small_blind?;
    next_after(active_order, sb)
}

fn next_after(active_order: &[u64], start: u64) -> Option<u64> {
    if active_order.is_empty() {
        return None;
    }
    let start_index = active_order.iter().position(|id| *id == start)?;
    for offset in 1..=active_order.len() {
        let idx = (start_index + offset) % active_order.len();
        let candidate = active_order[idx];
        if candidate != start {
            return Some(candidate);
        }
    }
    None
}

#[derive(Clone)]
struct PlayerState {
    starting_stack: f32,
    stack_remaining: f32,
    total_added_chips: f32,
    total_contribution: f32,
    cards: Vec<Card>,
    folded: bool,
    all_in: bool,
    sitting_out: bool,
}

impl PlayerState {
    fn new(player: &PlayerObj) -> Self {
        assert!(player.starting_stack.is_finite(), "Stacks must be finite");
        assert!(player.starting_stack >= 0.0, "Stacks cannot be negative");
        Self {
            starting_stack: player.starting_stack,
            stack_remaining: player.starting_stack,
            total_added_chips: 0.0,
            total_contribution: 0.0,
            cards: Vec::new(),
            folded: false,
            all_in: false,
            sitting_out: player.is_sitting_out.unwrap_or(false),
        }
    }

    fn is_available_for_action(&self) -> bool {
        !self.folded && !self.all_in && !self.sitting_out
    }
}

#[derive(Default)]
struct BettingRoundState {
    contributions: HashMap<u64, f32>,
    current_max: f32,
    /// The minimum raise amount (starts at big blind, increases with raises)
    min_raise: f32,
    /// The amount of the last raise (used to calculate min_raise)
    last_raise_amount: f32,
}

impl BettingRoundState {
    fn reset(&mut self, min_raise: f32) {
        self.contributions.clear();
        self.current_max = 0.0;
        self.min_raise = min_raise;
        self.last_raise_amount = min_raise;
    }

    fn record(&mut self, player_id: u64, amount: f32) -> f32 {
        let entry = self.contributions.entry(player_id).or_insert(0.0);
        *entry += amount;
        let previous_max = self.current_max;
        if *entry > self.current_max {
            self.current_max = *entry;
            // Update the minimum raise based on the raise amount
            let raise_amount = self.current_max - previous_max;
            if raise_amount > 0.0 {
                self.last_raise_amount = raise_amount;
                // In No-Limit, min_raise is at least the previous raise amount
                self.min_raise = self.min_raise.max(raise_amount);
            }
        }
        *entry
    }

    fn committed(&self, player_id: u64) -> f32 {
        *self.contributions.get(&player_id).unwrap_or(&0.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
    Unknown,
}

impl Street {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "preflop" => Street::Preflop,
            "flop" => Street::Flop,
            "turn" => Street::Turn,
            "river" => Street::River,
            "showdown" => Street::Showdown,
            _ => Street::Unknown,
        }
    }

    fn is_betting_round(self) -> bool {
        matches!(
            self,
            Street::Preflop | Street::Flop | Street::Turn | Street::River
        )
    }

    fn comes_after(self, other: Street) -> bool {
        matches!(
            (self, other),
            (Street::Flop, Street::Preflop)
                | (Street::Turn, Street::Preflop | Street::Flop)
                | (Street::River, Street::Preflop | Street::Flop | Street::Turn)
                | (
                    Street::Showdown,
                    Street::Preflop | Street::Flop | Street::Turn | Street::River
                )
        )
    }
}

struct BettingRotation {
    order: Vec<u64>,
    queue: VecDeque<u64>,
}

impl BettingRotation {
    fn new(order: Vec<u64>) -> Self {
        Self {
            order,
            queue: VecDeque::new(),
        }
    }

    fn start_round(
        &mut self,
        street: Street,
        dealer_id: u64,
        big_blind: Option<u64>,
        players: &HashMap<u64, PlayerState>,
    ) {
        if !street.is_betting_round() {
            self.queue.clear();
            return;
        }
        let start_player = match street {
            Street::Preflop => {
                if self.order.len() == 2 {
                    Some(dealer_id)
                } else {
                    big_blind.and_then(|bb| self.next_active_after(bb, players))
                }
            }
            Street::Flop | Street::Turn | Street::River => {
                self.next_active_after(dealer_id, players)
            }
            _ => None,
        };

        if let Some(start) = start_player {
            self.queue = self.build_queue_from(start, players);
        } else {
            self.queue.clear();
        }
    }

    fn expect_actor(&mut self, player_id: u64, players: &HashMap<u64, PlayerState>) {
        self.trim_inactive(players);
        let expected = self
            .queue
            .front()
            .copied()
            .expect("No players left to act in betting round");
        assert_eq!(
            expected, player_id,
            "Action out of turn: expected player {expected}, saw {player_id}"
        );
        self.queue.rotate_left(1);
        self.trim_inactive(players);
    }

    fn rebuild_after_raise(&mut self, raiser: u64, players: &HashMap<u64, PlayerState>) {
        if let Some(next) = self.next_active_after(raiser, players) {
            self.queue = self.build_queue_from(next, players);
        } else {
            self.queue.clear();
        }
    }

    fn remove_player(&mut self, player_id: u64) {
        self.queue.retain(|id| *id != player_id);
    }

    fn trim_inactive(&mut self, players: &HashMap<u64, PlayerState>) {
        while let Some(front) = self.queue.front() {
            if players
                .get(front)
                .map(|s| s.is_available_for_action())
                .unwrap_or(false)
            {
                break;
            }
            self.queue.pop_front();
        }
    }

    fn build_queue_from(&self, start: u64, players: &HashMap<u64, PlayerState>) -> VecDeque<u64> {
        let mut queue = VecDeque::new();
        if self.order.is_empty() {
            return queue;
        }
        let start_index = self.order.iter().position(|id| *id == start).unwrap_or(0);
        for offset in 0..self.order.len() {
            let idx = (start_index + offset) % self.order.len();
            let candidate = self.order[idx];
            if players
                .get(&candidate)
                .map(|s| s.is_available_for_action())
                .unwrap_or(false)
            {
                queue.push_back(candidate);
            }
        }
        queue
    }

    fn next_active_after(
        &self,
        player_id: u64,
        players: &HashMap<u64, PlayerState>,
    ) -> Option<u64> {
        if self.order.is_empty() {
            return None;
        }
        let start_index = self.order.iter().position(|id| *id == player_id)?;
        for offset in 1..=self.order.len() {
            let idx = (start_index + offset) % self.order.len();
            let candidate = self.order[idx];
            if players
                .get(&candidate)
                .map(|s| s.is_available_for_action())
                .unwrap_or(false)
            {
                return Some(candidate);
            }
        }
        None
    }
}

fn approx_eq(lhs: f32, rhs: f32) -> bool {
    // Use relative_eq with appropriate tolerances:
    // - epsilon: for values near zero, use f32::EPSILON as absolute tolerance
    // - max_relative: for larger values, allow 0.001% relative error to account
    //   for accumulated floating point errors in chip calculations
    relative_eq!(lhs, rhs, epsilon = f32::EPSILON, max_relative = 1e-5)
}

#[cfg(feature = "arena")]
struct HandHistoryArenaComparator<'a> {
    hh: &'a HandHistory,
    game_state: &'a crate::arena::GameState,
}

#[cfg(feature = "arena")]
impl<'a> HandHistoryArenaComparator<'a> {
    fn new(hh: &'a HandHistory, game_state: &'a crate::arena::GameState) -> Self {
        Self { hh, game_state }
    }

    fn assert_consistent(&self) {
        assert_eq!(
            self.hh.players.len(),
            self.game_state.num_players,
            "Hand history player count must match game state"
        );
        assert_eq!(
            self.hh.table_size as usize, self.game_state.num_players,
            "Hand history table size must match game state"
        );
        assert!(
            approx_eq(self.hh.small_blind_amount, self.game_state.small_blind),
            "Small blind mismatch"
        );
        assert!(
            approx_eq(self.hh.big_blind_amount, self.game_state.big_blind),
            "Big blind mismatch"
        );
        assert!(
            approx_eq(self.hh.ante_amount, self.game_state.ante),
            "Ante mismatch"
        );

        let board = collect_board_cards(self.hh);
        assert_eq!(board, self.game_state.board, "Board cards must align");

        for (idx, player) in self.hh.players.iter().enumerate() {
            let expected_stack = *self
                .game_state
                .starting_stacks
                .get(idx)
                .expect("Game state must include starting stack");
            assert!(
                approx_eq(player.starting_stack, expected_stack),
                "Starting stack mismatch for player {idx}"
            );
            let was_active = self.game_state.player_active.get(idx);
            if player.is_sitting_out.unwrap_or(false) {
                assert!(
                    !was_active,
                    "Player {idx} marked sitting out but active in game state"
                );
            }
        }

        let total_pot_hh: f32 = self.hh.pots.iter().map(|pot| pot.amount).sum();
        assert!(
            approx_eq(total_pot_hh, self.game_state.total_pot),
            "Total pot mismatch"
        );

        let mut hh_winnings = vec![0.0f32; self.game_state.num_players];
        for pot in &self.hh.pots {
            for win in &pot.player_wins {
                let idx = win.player_id as usize;
                hh_winnings[idx] += win.win_amount;
            }
        }
        for (idx, &amount) in self.game_state.player_winnings.iter().enumerate() {
            assert!(
                approx_eq(amount, hh_winnings[idx]),
                "Player {idx} winnings mismatch"
            );
        }
    }
}

fn collect_board_cards(hand_history: &HandHistory) -> Vec<Card> {
    let mut cards = Vec::new();
    for round in &hand_history.rounds {
        if let Some(round_cards) = &round.cards {
            cards.extend(round_cards.iter().copied());
        }
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Card, Suit, Value};
    use crate::open_hand_history::{BetLimitObj, BetType, GameType, PlayerWinsObj};

    fn sample_hand_history() -> HandHistory {
        let players = vec![
            PlayerObj {
                id: 0,
                seat: 1,
                name: "P1".into(),
                display: None,
                starting_stack: 100.0,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
            PlayerObj {
                id: 1,
                seat: 2,
                name: "P2".into(),
                display: None,
                starting_stack: 100.0,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
        ];

        let deal_p1 = ActionObj {
            action_number: 1,
            player_id: 0,
            action: Action::DealtCards,
            amount: 0.0,
            is_allin: false,
            cards: Some(vec![
                Card::new(Value::Ace, Suit::Spade),
                Card::new(Value::King, Suit::Heart),
            ]),
        };
        let deal_p2 = ActionObj {
            action_number: 2,
            player_id: 1,
            action: Action::DealtCards,
            amount: 0.0,
            is_allin: false,
            cards: Some(vec![
                Card::new(Value::Queen, Suit::Club),
                Card::new(Value::Queen, Suit::Diamond),
            ]),
        };
        let post_sb = ActionObj {
            action_number: 3,
            player_id: 0,
            action: Action::PostSmallBlind,
            amount: 1.0,
            is_allin: false,
            cards: None,
        };
        let post_bb = ActionObj {
            action_number: 4,
            player_id: 1,
            action: Action::PostBigBlind,
            amount: 2.0,
            is_allin: false,
            cards: None,
        };
        let fold = ActionObj {
            action_number: 5,
            player_id: 0,
            action: Action::Fold,
            amount: 0.0,
            is_allin: false,
            cards: None,
        };

        let rounds = vec![RoundObj {
            id: 1,
            street: "Preflop".into(),
            cards: None,
            actions: vec![deal_p1, deal_p2, post_sb, post_bb, fold],
        }];

        let pots = vec![PotObj {
            number: 1,
            amount: 3.0,
            rake: None,
            jackpot: None,
            player_wins: vec![PlayerWinsObj {
                player_id: 1,
                win_amount: 3.0,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            }],
        }];

        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "rs_poker".into(),
            network_name: "rs_poker".into(),
            internal_version: "test".into(),
            tournament: false,
            tournament_info: None,
            game_number: "1".into(),
            start_date_utc: None,
            table_name: "table".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        }
    }

    #[test]
    fn valid_history_passes() {
        let history = sample_hand_history();
        assert_valid_open_hand_history(&history);
    }

    #[test]
    #[should_panic]
    fn duplicate_card_panics() {
        let mut history = sample_hand_history();
        if let Some(round) = history.rounds.first_mut()
            && let Some(action) = round
                .actions
                .iter_mut()
                .find(|a| a.player_id == 1 && matches!(a.action, Action::DealtCards))
        {
            action.cards = Some(vec![
                Card::new(Value::Ace, Suit::Spade),
                Card::new(Value::Two, Suit::Club),
            ]);
        }
        assert_valid_open_hand_history(&history);
    }

    #[test]
    fn allows_call_of_tiny_all_in_bet() {
        let players = vec![
            PlayerObj {
                id: 0,
                seat: 1,
                name: "Caller".into(),
                display: None,
                starting_stack: 100.0,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
            PlayerObj {
                id: 1,
                seat: 2,
                name: "Shorty".into(),
                display: None,
                starting_stack: 1.0005,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
        ];

        let preflop_actions = vec![
            ActionObj {
                action_number: 1,
                player_id: 0,
                action: Action::DealtCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Ten, Suit::Spade),
                    Card::new(Value::Nine, Suit::Heart),
                ]),
            },
            ActionObj {
                action_number: 2,
                player_id: 1,
                action: Action::DealtCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Eight, Suit::Club),
                    Card::new(Value::Seven, Suit::Diamond),
                ]),
            },
            ActionObj {
                action_number: 3,
                player_id: 0,
                action: Action::PostSmallBlind,
                amount: 1.0,
                is_allin: false,
                cards: None,
            },
            ActionObj {
                action_number: 4,
                player_id: 1,
                action: Action::PostBigBlind,
                amount: 1.0,
                is_allin: false,
                cards: None,
            },
            ActionObj {
                action_number: 5,
                player_id: 0,
                action: Action::Check,
                amount: 0.0,
                is_allin: false,
                cards: None,
            },
            ActionObj {
                action_number: 6,
                player_id: 1,
                action: Action::Check,
                amount: 0.0,
                is_allin: false,
                cards: None,
            },
        ];

        let flop_actions = vec![
            ActionObj {
                action_number: 1,
                player_id: 1,
                action: Action::Bet,
                amount: 0.0005,
                is_allin: true,
                cards: None,
            },
            ActionObj {
                action_number: 2,
                player_id: 0,
                action: Action::Call,
                amount: 0.0005,
                is_allin: false,
                cards: None,
            },
        ];

        let showdown_actions = vec![ActionObj {
            action_number: 1,
            player_id: 0,
            action: Action::ShowsCards,
            amount: 0.0,
            is_allin: false,
            cards: Some(vec![
                Card::new(Value::Ten, Suit::Spade),
                Card::new(Value::Nine, Suit::Heart),
            ]),
        }];

        let rounds = vec![
            RoundObj {
                id: 1,
                street: "Preflop".into(),
                cards: None,
                actions: preflop_actions,
            },
            RoundObj {
                id: 2,
                street: "Flop".into(),
                cards: Some(vec![
                    Card::new(Value::Two, Suit::Club),
                    Card::new(Value::Five, Suit::Heart),
                    Card::new(Value::Jack, Suit::Diamond),
                ]),
                actions: flop_actions,
            },
            RoundObj {
                id: 3,
                street: "Showdown".into(),
                cards: None,
                actions: showdown_actions,
            },
        ];

        let pots = vec![PotObj {
            number: 1,
            amount: 2.001,
            rake: None,
            jackpot: None,
            player_wins: vec![PlayerWinsObj {
                player_id: 0,
                win_amount: 2.001,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            }],
        }];

        let history = HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "rs_poker".into(),
            network_name: "rs_poker".into(),
            internal_version: "test".into(),
            tournament: false,
            tournament_info: None,
            game_number: "micro-call".into(),
            start_date_utc: None,
            table_name: "tiny-pot".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 1.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        };

        assert_valid_open_hand_history(&history);
    }

    #[test]
    fn allows_short_stack_all_in_bet() {
        let history = HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "rs_poker".into(),
            network_name: "rs_poker_arena".into(),
            internal_version: "test".into(),
            tournament: false,
            tournament_info: None,
            game_number: "short_stack".into(),
            start_date_utc: None,
            table_name: "table".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 1.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![
                PlayerObj {
                    id: 0,
                    seat: 1,
                    name: "Deep".into(),
                    display: None,
                    starting_stack: 10.0,
                    player_bounty: None,
                    is_sitting_out: Some(false),
                },
                PlayerObj {
                    id: 1,
                    seat: 2,
                    name: "Shorty".into(),
                    display: None,
                    starting_stack: 1.0005,
                    player_bounty: None,
                    is_sitting_out: Some(false),
                },
            ],
            rounds: vec![
                RoundObj {
                    id: 1,
                    street: "Preflop".into(),
                    cards: None,
                    actions: vec![
                        ActionObj {
                            action_number: 1,
                            player_id: 0,
                            action: Action::DealtCards,
                            amount: 0.0,
                            is_allin: false,
                            cards: Some(vec![
                                Card::new(Value::Ace, Suit::Spade),
                                Card::new(Value::King, Suit::Heart),
                            ]),
                        },
                        ActionObj {
                            action_number: 2,
                            player_id: 1,
                            action: Action::DealtCards,
                            amount: 0.0,
                            is_allin: false,
                            cards: Some(vec![
                                Card::new(Value::Queen, Suit::Club),
                                Card::new(Value::Jack, Suit::Diamond),
                            ]),
                        },
                        ActionObj {
                            action_number: 3,
                            player_id: 0,
                            action: Action::PostSmallBlind,
                            amount: 1.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 4,
                            player_id: 1,
                            action: Action::PostBigBlind,
                            amount: 1.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 5,
                            player_id: 0,
                            action: Action::Check,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 6,
                            player_id: 1,
                            action: Action::Check,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                    ],
                },
                RoundObj {
                    id: 2,
                    street: "Flop".into(),
                    cards: Some(vec![
                        Card::new(Value::Two, Suit::Club),
                        Card::new(Value::Five, Suit::Diamond),
                        Card::new(Value::Seven, Suit::Heart),
                    ]),
                    actions: vec![
                        ActionObj {
                            action_number: 1,
                            player_id: 1,
                            action: Action::Check,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 2,
                            player_id: 0,
                            action: Action::Check,
                            amount: 0.0,
                            is_allin: false,
                            cards: None,
                        },
                    ],
                },
                RoundObj {
                    id: 3,
                    street: "Turn".into(),
                    cards: Some(vec![Card::new(Value::Nine, Suit::Spade)]),
                    actions: vec![
                        ActionObj {
                            action_number: 1,
                            player_id: 1,
                            action: Action::Bet,
                            amount: 0.0005,
                            is_allin: true,
                            cards: None,
                        },
                        ActionObj {
                            action_number: 2,
                            player_id: 0,
                            action: Action::Raise,
                            amount: 9.0,
                            is_allin: true,
                            cards: None,
                        },
                    ],
                },
            ],
            pots: vec![PotObj {
                number: 1,
                amount: 11.0005,
                rake: None,
                jackpot: None,
                player_wins: vec![PlayerWinsObj {
                    player_id: 0,
                    win_amount: 11.0005,
                    cashout_amount: None,
                    cashout_fee: None,
                    bonus_amount: None,
                    contributed_rake: None,
                }],
            }],
            tournament_bounties: None,
        };

        assert_valid_open_hand_history(&history);
    }

    #[test]
    fn allows_tiny_all_in_raise_after_call() {
        // Regression test for fuzzer crash: when a player has posted a large blind
        // and only has a tiny amount remaining, going all-in with that tiny amount
        // should be accepted as a valid raise even though the increase is smaller
        // than f32::EPSILON. This tests floating point precision with large bet sizes.
        let players = vec![
            PlayerObj {
                id: 0,
                seat: 1,
                name: "SB".into(),
                display: None,
                starting_stack: 195.26274,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
            PlayerObj {
                id: 1,
                seat: 2,
                name: "BB".into(),
                display: None,
                starting_stack: 195.26282,
                player_bounty: None,
                is_sitting_out: Some(false),
            },
        ];

        let preflop_actions = vec![
            ActionObj {
                action_number: 1,
                player_id: 0,
                action: Action::DealtCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Three, Suit::Diamond),
                    Card::new(Value::Five, Suit::Heart),
                ]),
            },
            ActionObj {
                action_number: 2,
                player_id: 1,
                action: Action::DealtCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Five, Suit::Diamond),
                    Card::new(Value::Ace, Suit::Heart),
                ]),
            },
            ActionObj {
                action_number: 3,
                player_id: 0,
                action: Action::PostSmallBlind,
                amount: 195.24321,
                is_allin: false,
                cards: None,
            },
            ActionObj {
                action_number: 4,
                player_id: 1,
                action: Action::PostBigBlind,
                amount: 195.26271,
                is_allin: false,
                cards: None,
            },
            // SB calls to match BB
            ActionObj {
                action_number: 5,
                player_id: 0,
                action: Action::Call,
                amount: 0.019500732,
                is_allin: false,
                cards: None,
            },
            // BB goes all-in with tiny remaining amount - this is the key action
            // The raise amount (0.00010681152) is smaller than f32::EPSILON
            // but should be accepted because it's an all-in
            ActionObj {
                action_number: 6,
                player_id: 1,
                action: Action::Raise,
                amount: 0.00010681152,
                is_allin: true,
                cards: None,
            },
        ];

        let showdown_actions = vec![
            ActionObj {
                action_number: 1,
                player_id: 0,
                action: Action::ShowsCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Three, Suit::Diamond),
                    Card::new(Value::Five, Suit::Heart),
                ]),
            },
            ActionObj {
                action_number: 2,
                player_id: 1,
                action: Action::ShowsCards,
                amount: 0.0,
                is_allin: false,
                cards: Some(vec![
                    Card::new(Value::Five, Suit::Diamond),
                    Card::new(Value::Ace, Suit::Heart),
                ]),
            },
        ];

        let rounds = vec![
            RoundObj {
                id: 1,
                street: "Preflop".into(),
                cards: None,
                actions: preflop_actions,
            },
            RoundObj {
                id: 2,
                street: "Flop".into(),
                cards: Some(vec![
                    Card::new(Value::Three, Suit::Spade),
                    Card::new(Value::Six, Suit::Heart),
                    Card::new(Value::Jack, Suit::Heart),
                ]),
                actions: vec![],
            },
            RoundObj {
                id: 3,
                street: "Turn".into(),
                cards: Some(vec![Card::new(Value::Nine, Suit::Diamond)]),
                actions: vec![],
            },
            RoundObj {
                id: 4,
                street: "River".into(),
                cards: Some(vec![Card::new(Value::Two, Suit::Club)]),
                actions: vec![],
            },
            RoundObj {
                id: 5,
                street: "Showdown".into(),
                cards: None,
                actions: showdown_actions,
            },
        ];

        // Total pot: SB contributed 195.26271 + BB contributed 195.26282 = 390.52553
        let total_pot = 195.26271 + 0.00010681152 + 195.24321 + 0.019500732;
        let pots = vec![PotObj {
            number: 1,
            amount: total_pot,
            rake: None,
            jackpot: None,
            player_wins: vec![PlayerWinsObj {
                player_id: 1,
                win_amount: total_pot,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            }],
        }];

        let history = HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "rs_poker".into(),
            network_name: "rs_poker_arena".into(),
            internal_version: "test".into(),
            tournament: false,
            tournament_info: None,
            game_number: "tiny_raise".into(),
            start_date_utc: None,
            table_name: "table".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0.0,
            }),
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 195.24321,
            big_blind_amount: 195.26271,
            ante_amount: 0.0,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        };

        assert_valid_open_hand_history(&history);
    }
}
