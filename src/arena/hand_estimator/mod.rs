//! Opponent hand-distribution estimation for CFR.
//!
//! A `HandDistributionEstimator` guesses, from the perspective of the seat
//! that is about to act, the hole-card distribution of every *other* live
//! player. The CFR engine calls it once per `act()` and then samples concrete
//! worlds from the result (see `world::sample_world`). This lets the solver
//! play against ranges instead of the pinned, fully-known hands.

