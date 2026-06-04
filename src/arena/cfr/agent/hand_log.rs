//! Copy-on-write action log for CFR hand-history estimation.
//!
//! The real hand is frozen ONCE at depth 0 into a shared immutable `prefix`;
//! each simulation's own actions are an owned `tail`, copied on descent so a
//! child sees the full betting path that produced its state. No mutable buffer
//! is ever shared across concurrent tasks: the `prefix` is read-only and each
//! simulation owns its `tail`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use smallvec::SmallVec;

use crate::arena::GameState;
use crate::arena::action::Action;
use crate::arena::historian::{Historian, HistorianError, HistorianLock};

/// Inline capacity for one line's appended actions. The real hand lives in the
/// shared `prefix`, so a tail only holds a single recursion line's continuation
/// (short in practice). Longer lines spill to the heap.
const INLINE: usize = 16;

/// One line of play's action log: a shared immutable prefix plus an owned tail.
///
/// `Clone` is a shallow Arc clone: a clone SHARES the same `tail` as the
/// original — both observe each other's `record`s. This is how the per-sim
/// `HandLogHistorian` (the writer) and the agents (readers) share one tail.
/// To start a *new, independent* line of play (a child sub-simulation), use
/// [`HandLog::spawn_child`], which forks a fresh tail.
#[derive(Clone)]
pub struct HandLog {
    prefix: Arc<[Action]>,
    tail: Arc<Mutex<SmallVec<[Action; INLINE]>>>,
}

#[allow(dead_code)]
impl HandLog {
    /// Empty prefix, empty tail. Used for a depth-0 top-level simulation before
    /// its real hand has been recorded.
    pub fn new() -> Self {
        Self {
            prefix: Arc::from([] as [Action; 0]),
            tail: Arc::new(Mutex::new(SmallVec::new())),
        }
    }

    /// Append one action to this line's tail. Stores the `Action` only — no
    /// `GameState` clone. Panics on a poisoned lock (callers are single-sim).
    pub fn record(&self, action: Action) {
        self.tail
            .lock()
            .expect("HandLog tail poisoned")
            .push(action);
    }

    /// The full ordered action sequence: shared prefix then this line's tail.
    /// Materialized into an owned `Vec` so it can back a `GameLog` across the
    /// async `estimate` boundary without holding the lock.
    pub fn to_actions(&self) -> Vec<Action> {
        let tail = self.tail.lock().expect("HandLog tail poisoned");
        let mut out = Vec::with_capacity(self.prefix.len() + tail.len());
        out.extend_from_slice(&self.prefix);
        out.extend(tail.iter().cloned());
        out
    }

    /// Collapse `prefix + tail` into a single shared immutable prefix with an
    /// empty tail. Called ONCE at depth 0 to absorb the real hand so
    /// descendants never re-copy it.
    pub fn freeze(&self) -> HandLog {
        HandLog {
            prefix: Arc::from(self.to_actions()),
            tail: Arc::new(Mutex::new(SmallVec::new())),
        }
    }

    /// A child log for a spawned sub-simulation: the same shared `prefix`, and a
    /// fresh tail seeded with a copy of this line's accumulated tail. Only the
    /// short simulated tail is copied; the real-hand prefix is shared by Arc.
    pub fn spawn_child(&self) -> HandLog {
        let seed = self.tail.lock().expect("HandLog tail poisoned").clone();
        HandLog {
            prefix: self.prefix.clone(),
            tail: Arc::new(Mutex::new(seed)),
        }
    }
}

impl Default for HandLog {
    fn default() -> Self {
        Self::new()
    }
}

/// The single per-simulation writer appending each recorded action into a
/// shared [`HandLog`]. Lightweight: holds one `HandLog` (two Arcs).
#[allow(dead_code)]
pub struct HandLogHistorian {
    log: HandLog,
}

#[allow(dead_code)]
impl HandLogHistorian {
    pub fn new(log: HandLog) -> Self {
        Self { log }
    }
}

#[async_trait]
impl Historian for HandLogHistorian {
    async fn record_action(
        &mut self,
        _id: u128,
        _game_state: &GameState,
        action: &Action,
    ) -> Result<(), HistorianError> {
        let mut tail = self
            .log
            .tail
            .lock()
            .map_err(|_| HistorianError::LockPoisoned {
                lock: HistorianLock::HandLog,
            })?;
        tail.push(action.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::game_state::Round;
    use crate::core::Card;

    fn ra(r: Round) -> Action {
        Action::RoundAdvance(r)
    }
    fn deal(n: u8) -> Action {
        Action::DealCommunity(Card::from(n))
    }

    #[test]
    fn new_is_empty() {
        assert!(HandLog::new().to_actions().is_empty());
    }

    #[test]
    fn record_appends_in_order() {
        let log = HandLog::new();
        log.record(ra(Round::Preflop));
        log.record(deal(10));
        assert_eq!(log.to_actions(), vec![ra(Round::Preflop), deal(10)]);
    }

    #[test]
    fn freeze_collapses_into_prefix_and_empties_tail() {
        let log = HandLog::new();
        log.record(ra(Round::Preflop));
        log.record(deal(10));
        let frozen = log.freeze();
        // Frozen sees the same sequence...
        assert_eq!(frozen.to_actions(), vec![ra(Round::Preflop), deal(10)]);
        // ...but its tail is empty: new records on the frozen log start fresh,
        // and the original is unaffected.
        frozen.record(deal(20));
        assert_eq!(
            frozen.to_actions(),
            vec![ra(Round::Preflop), deal(10), deal(20)]
        );
        assert_eq!(log.to_actions(), vec![ra(Round::Preflop), deal(10)]);
    }

    #[test]
    fn spawn_child_copies_tail_and_is_independent() {
        let parent = HandLog::new();
        parent.record(ra(Round::Flop));
        let child = parent.spawn_child();
        // Child starts from the parent's accumulated tail.
        assert_eq!(child.to_actions(), vec![ra(Round::Flop)]);
        // Appends to each are independent (no shared mutable tail).
        child.record(deal(30));
        parent.record(deal(40));
        assert_eq!(child.to_actions(), vec![ra(Round::Flop), deal(30)]);
        assert_eq!(parent.to_actions(), vec![ra(Round::Flop), deal(40)]);
    }

    #[test]
    fn full_path_through_freeze_then_two_descents() {
        // Depth 0: real hand recorded, then frozen.
        let root = HandLog::new();
        root.record(ra(Round::Preflop));
        let d0 = root.freeze(); // prefix = [Preflop], tail = []

        // Depth 1: child starts empty, accumulates its line.
        let d1 = d0.spawn_child();
        d1.record(deal(1));

        // Depth 2: child copies d1's tail, then accumulates.
        let d2 = d1.spawn_child();
        d2.record(deal(2));

        // Full path = real hand + depth-1 line + depth-2 line.
        assert_eq!(d2.to_actions(), vec![ra(Round::Preflop), deal(1), deal(2)]);
    }

    #[tokio::test]
    async fn historian_records_into_shared_log() {
        use crate::arena::Historian;

        let log = HandLog::new();
        let mut hist = HandLogHistorian::new(log.clone());

        let game_state = crate::arena::GameStateBuilder::default()
            .num_players_with_stack(2, 100.0)
            .big_blind(2.0)
            .build()
            .unwrap();

        hist.record_action(0, &game_state, &ra(Round::Preflop))
            .await
            .unwrap();
        hist.record_action(0, &game_state, &deal(7))
            .await
            .unwrap();

        // The historian's clone shares the tail with `log`, so the appends are
        // visible through the original handle.
        assert_eq!(log.to_actions(), vec![ra(Round::Preflop), deal(7)]);
    }
}
