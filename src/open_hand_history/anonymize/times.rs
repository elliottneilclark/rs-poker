//! Timestamp fuzzing.
//!
//! Hand histories frequently include exact UTC timestamps, and those
//! timestamps are enough to cross-reference a hand against public
//! tournament schedules, leaderboards, or shared screenshots. The
//! [`TimeFuzzer`] obscures this by combining two noise sources:
//!
//! 1. A **global shift** chosen once at construction and applied to
//!    every timestamp. This shifts the whole stream by the same
//!    amount so the session "drifts" to a different wall-clock window
//!    while internal relative ordering is preserved.
//! 2. A **per-hand jitter** added on top of the global shift for each
//!    hand's `start_date_utc`. This breaks up the exact inter-hand
//!    spacing that could otherwise be used to fingerprint a session.
//!
//! Tournament `start_date_utc` receives only the global shift (no
//! jitter) so that every hand in the same tournament continues to
//! report the same tournament start time.
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use rand::RngExt;
use rand::rngs::StdRng;
use std::time::Duration;

/// Applies a global shift plus per-hand jitter to `DateTime<Utc>`
/// values.
pub struct TimeFuzzer {
    global_shift: ChronoDuration,
    max_jitter_ms: i64,
}

impl TimeFuzzer {
    /// Pick a random global shift and remember the jitter bound.
    ///
    /// `max_global_shift` and `max_per_hand_jitter` are treated as
    /// symmetric bounds: the actual shift is uniform in
    /// `[-max_global_shift, +max_global_shift]`, and jitter on each
    /// hand is uniform in `[-max_per_hand_jitter, +max_per_hand_jitter]`.
    pub fn new(
        max_global_shift: Duration,
        max_per_hand_jitter: Duration,
        rng: &mut StdRng,
    ) -> Self {
        let max_shift_ms = duration_to_i64_ms(max_global_shift);
        let shift_ms = if max_shift_ms == 0 {
            0
        } else {
            rng.random_range(-max_shift_ms..=max_shift_ms)
        };
        Self {
            global_shift: ChronoDuration::milliseconds(shift_ms),
            max_jitter_ms: duration_to_i64_ms(max_per_hand_jitter),
        }
    }

    /// Apply only the global shift (no jitter).
    ///
    /// Use this for values that should stay constant across related
    /// hands, like a tournament's start date.
    pub fn shift_only(&self, t: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
        t.map(|dt| dt + self.global_shift)
    }

    /// Apply the global shift plus a fresh per-hand jitter.
    pub fn shift_with_jitter(
        &self,
        t: Option<DateTime<Utc>>,
        rng: &mut StdRng,
    ) -> Option<DateTime<Utc>> {
        t.map(|dt| dt + self.global_shift + self.draw_jitter(rng))
    }

    fn draw_jitter(&self, rng: &mut StdRng) -> ChronoDuration {
        if self.max_jitter_ms == 0 {
            ChronoDuration::zero()
        } else {
            ChronoDuration::milliseconds(rng.random_range(-self.max_jitter_ms..=self.max_jitter_ms))
        }
    }
}

/// Saturating conversion from [`std::time::Duration`] to `i64`
/// milliseconds.
///
/// `Duration::as_millis` returns `u128`; we clamp to `i64::MAX` to
/// keep downstream arithmetic infallible. Values large enough to
/// saturate would not represent plausible fuzzing windows anyway.
fn duration_to_i64_ms(d: Duration) -> i64 {
    let ms = d.as_millis();
    if ms > i64::MAX as u128 {
        i64::MAX
    } else {
        ms as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(11)
    }

    fn t(s: &str) -> Option<DateTime<Utc>> {
        Some(DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc))
    }

    #[test]
    fn zero_config_is_identity() {
        let fuzzer = TimeFuzzer::new(Duration::ZERO, Duration::ZERO, &mut rng());
        let input = t("2024-01-01T12:00:00Z");
        let mut r = rng();
        assert_eq!(fuzzer.shift_only(input), input);
        assert_eq!(fuzzer.shift_with_jitter(input, &mut r), input);
    }

    #[test]
    fn shift_only_is_consistent_across_calls() {
        let fuzzer = TimeFuzzer::new(Duration::from_secs(60 * 60), Duration::ZERO, &mut rng());
        let a = fuzzer.shift_only(t("2024-01-01T12:00:00Z"));
        let b = fuzzer.shift_only(t("2024-01-01T12:00:00Z"));
        assert_eq!(a, b);
    }

    #[test]
    fn jitter_stays_within_configured_bound() {
        let max = Duration::from_secs(5);
        let fuzzer = TimeFuzzer::new(Duration::ZERO, max, &mut rng());
        let input = t("2024-01-01T12:00:00Z").unwrap();
        let mut r = rng();
        for _ in 0..100 {
            let out = fuzzer.shift_with_jitter(Some(input), &mut r).unwrap();
            let delta = (out - input).num_milliseconds().abs();
            assert!(delta <= max.as_millis() as i64);
        }
    }

    #[test]
    fn shift_affects_every_input_by_same_amount() {
        let fuzzer = TimeFuzzer::new(Duration::from_secs(60 * 60), Duration::ZERO, &mut rng());
        let a_in = t("2024-01-01T12:00:00Z").unwrap();
        let b_in = t("2024-01-01T13:30:00Z").unwrap();
        let a_out = fuzzer.shift_only(Some(a_in)).unwrap();
        let b_out = fuzzer.shift_only(Some(b_in)).unwrap();
        assert_eq!(b_in - a_in, b_out - a_out);
    }

    #[test]
    fn none_passes_through() {
        let fuzzer = TimeFuzzer::new(Duration::from_secs(60), Duration::from_secs(1), &mut rng());
        assert_eq!(fuzzer.shift_only(None), None);
        let mut r = rng();
        assert_eq!(fuzzer.shift_with_jitter(None, &mut r), None);
    }
}
