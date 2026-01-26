//! This module provides the ability to simulate a mutli table independent chip
//! tournament. It does this via simulation. Different heros and villans go to
//! all in show downs. Then the resulting placements are computed as each player
//! busts.
//!
//! This method does not require a recursive dive N! so it makes simulating
//! tournaments with many different people and different payments feasible.
//! However it comes with some downsides.
//!
//! - The results are not repeatable.
//! - Small SNG's would be faster to compute with full ICM rather than
//!   simulations
//!
//! However it does have some other nice properties
//!
//! - It's parrallelizable. This can be farmed out to many different cores to
//!   speed
//! this up. Since each tournament is indepent there's little coordination
//! oeverhead needed.
//! - We can change the players skill easily. Since ICM just looks at the
//!   percentage or outstanding chips
use rand::{Rng, rng, seq::SliceRandom};

/// Simulate a tournament by running a series of all
/// in showdowns. This helps deterimine the value of each
/// chip stack in a tournament with payout schedules.
///
///
/// # Arguments
///
/// * `chip_stacks` - The chip stacks of each player in the tournament.
/// * `payments` - The payout schedule for the tournament.
pub fn simulate_icm_tournament(chip_stacks: &[i32], payments: &[i32]) -> Vec<i32> {
    // We're going to mutate in place so move the chip stacks into a mutable vector.
    let mut remaining_stacks: Vec<i32> = chip_stacks.into();
    // Thread local rng.
    let mut rng = rng();
    // Which place in the next player to bust will get.
    let mut next_place = remaining_stacks.len() - 1;

    // The results.
    let mut winnings = vec![0; remaining_stacks.len()];
    // set all the players as still having chips remaining.
    let mut remaining_players: Vec<usize> = (0..chip_stacks.len()).collect();

    while !remaining_players.is_empty() {
        // Shuffle the players because we are going to use
        // the last two in the vector.
        // That allows O(1) pop and then usually push
        remaining_players.shuffle(&mut rng);

        // While this looks like it should be a ton of
        // mallocing and free-ing memory
        // because the vector never grows and ususally stays
        // the same size, it's remarkably fast.
        let hero = remaining_players.pop().expect("There should always be one");

        // If there are two players remaining then run the game
        if let Some(villan) = remaining_players.pop() {
            // For now assume that each each player has the same skill.
            // TODO: Check to see if adding in a skill(running avg of win %) array for each
            // player is needed.
            let hero_won: bool = rng.random_bool(0.5);

            // can't bet chips that can't be called.
            let effective_stacks = remaining_stacks[hero].min(remaining_stacks[villan]);
            let hero_change: i32 = if hero_won {
                effective_stacks
            } else {
                -effective_stacks
            };
            remaining_stacks[hero] += hero_change;
            remaining_stacks[villan] -= hero_change;

            // Check if hero was eliminated.
            if remaining_stacks[hero] == 0 {
                if next_place < payments.len() {
                    winnings[hero] = payments[next_place];
                }
                next_place -= 1;
            } else {
                remaining_players.push(hero);
            }

            // Now check if the villan was eliminated.
            if remaining_stacks[villan] == 0 {
                if next_place < payments.len() {
                    winnings[villan] = payments[next_place];
                }
                next_place -= 1;
            } else {
                remaining_players.push(villan);
            }
        } else {
            // If there's only a hero and no
            // villan then give the hero the money
            //
            // They have earned it.
            winnings[hero] = payments[next_place];
        };
    }
    winnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_players_works() {
        let payments = vec![10_000, 6_000, 4_000, 1_000, 800];
        let mut rng = rng();

        for num_players in &[2, 3, 4, 5, 15, 16, 32] {
            let chips: Vec<i32> = (0..*num_players)
                .map(|_pn| rng.random_range(1..500))
                .collect();

            let _res = simulate_icm_tournament(&chips, &payments);
        }
    }
    #[test]
    fn test_huge_lead_wins() {
        let stacks = vec![1000, 2, 1];
        let payments = vec![100, 30, 10];

        let mut total_winnings = vec![0; 3];
        let num_trials = 1000;

        for _i in 0..num_trials {
            let single_wins = simulate_icm_tournament(&stacks, &payments);
            total_winnings = total_winnings
                .iter()
                .zip(single_wins.iter())
                .map(|(a, b)| a + b)
                .collect()
        }

        let final_share: Vec<f64> = total_winnings
            .iter()
            .map(|v| f64::from(*v) / f64::from(num_trials))
            .collect();

        assert!(
            final_share[0] > final_share[1],
            "The total winnings of a player with most of the chips should be above the rest."
        );
    }

    #[test]
    fn about_same() {
        let stacks = vec![1000, 1000, 999];
        let payments = vec![100, 30, 10];

        let mut total_winnings = vec![0; 3];
        let num_trials = 1000;

        for _i in 0..num_trials {
            let single_wins = simulate_icm_tournament(&stacks, &payments);
            total_winnings = total_winnings
                .iter()
                .zip(single_wins.iter())
                .map(|(a, b)| a + b)
                .collect();
        }

        let final_share: Vec<f64> = total_winnings
            .iter()
            .map(|v| f64::from(*v) / f64::from(num_trials))
            .collect();

        let sum: f64 = final_share.iter().sum();
        let avg = sum / (final_share.len() as f64);

        for &share in final_share.iter() {
            assert!(share < 1.1 * avg);
            assert!(1.1 * share > avg);
        }
    }

    /// Verifies that total winnings equals the total payment pool.
    #[test]
    fn test_total_winnings_consistent() {
        let payments = vec![100, 30, 10];

        // Run many trials with the same number of players as payments
        let stacks = vec![100, 100, 100]; // 3 players, 3 payment positions
        let total_pool: i32 = payments.iter().sum();

        for _ in 0..100 {
            let winnings = simulate_icm_tournament(&stacks, &payments);
            let total_winnings: i32 = winnings.iter().sum();
            assert_eq!(
                total_winnings, total_pool,
                "Total winnings should equal total payment pool when players == payments"
            );
        }

        // Test with fewer players than payments - only top N payments used
        let stacks_2 = vec![100, 100]; // 2 players
        let expected_pool_2: i32 = payments[0] + payments[1]; // 1st and 2nd place

        for _ in 0..100 {
            let winnings = simulate_icm_tournament(&stacks_2, &payments);
            let total_winnings: i32 = winnings.iter().sum();
            assert_eq!(
                total_winnings, expected_pool_2,
                "With 2 players, should get 1st and 2nd place payouts"
            );
        }
    }

    /// Verifies that each payment amount is awarded to exactly one player.
    #[test]
    fn test_all_players_get_payout_when_enough_payments() {
        let stacks = vec![100, 100, 100];
        let payments = vec![100, 50, 25];

        for _ in 0..100 {
            let winnings = simulate_icm_tournament(&stacks, &payments);

            // Each player should receive exactly one payout
            let mut received_payouts: Vec<i32> = winnings.to_vec();
            received_payouts.sort();
            let mut expected_payouts = payments.clone();
            expected_payouts.sort();

            assert_eq!(
                received_payouts, expected_payouts,
                "Each payment should go to exactly one player"
            );
        }
    }

    /// Verifies that only paid positions receive payouts when there are more players than payment slots.
    #[test]
    fn test_more_players_than_payments() {
        let stacks = vec![100, 100, 100, 100, 100];
        let payments = vec![100, 50]; // Only top 2 get paid

        for _ in 0..100 {
            let winnings = simulate_icm_tournament(&stacks, &payments);

            // Count how many players got paid
            let paid_count = winnings.iter().filter(|&&w| w > 0).count();
            assert_eq!(
                paid_count, 2,
                "Only 2 players should be paid when payments has 2 entries"
            );

            // Total should still match
            let total: i32 = winnings.iter().sum();
            assert_eq!(total, 150, "Total should match payment pool");
        }
    }

    /// Test two-player case specifically.
    /// This directly tests chip transfer logic (lines 70, 72).
    #[test]
    fn test_two_player_winner_takes_all() {
        let stacks = vec![100, 50];
        let payments = vec![100, 0];

        for _ in 0..100 {
            let winnings = simulate_icm_tournament(&stacks, &payments);

            // Exactly one player should win 100
            assert!(
                (winnings[0] == 100 && winnings[1] == 0)
                    || (winnings[0] == 0 && winnings[1] == 100),
                "One player should win all, got: {:?}",
                winnings
            );
        }
    }

    /// Test that the winner with massive chip lead almost always wins first place.
    /// This tests that chip transfers work correctly (hero_change sign matters).
    #[test]
    fn test_dominant_chip_leader_wins_first() {
        let stacks = vec![10000, 1, 1];
        let payments = vec![100, 0, 0];

        let mut player0_first_count = 0;
        let trials = 1000;

        for _ in 0..trials {
            let winnings = simulate_icm_tournament(&stacks, &payments);
            if winnings[0] == 100 {
                player0_first_count += 1;
            }
        }

        // Player 0 should win first place in the vast majority of simulations
        assert!(
            player0_first_count > 950,
            "Chip leader should win almost always, but only won {}/{}",
            player0_first_count,
            trials
        );
    }
}
