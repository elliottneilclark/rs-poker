//! This module provides the ability to simulate a mutli table independent chip tournament.
//! It does this via simulation. Different heros and villans go to all in show downs. Then
//! the resulting placements are computed as each player busts.
//!
//! This method does not require a recursive dive N! so it makes simulating
//! tournaments with many different people and different payments feasible. However it comes with
//! some downsides.
//!
//! - The results are not repeatable.
//! - Small SNG's would be faster to compute with full ICM rather than simulations
//!
//! However it does have some other nice properties
//!
//! - It's parrallelizable. This can be farmed out to many different cores to speed
//! this up. Since each tournament is indepent there's little coordination oeverhead needed.
//! - We can change the players skill easily. Since ICM just looks at the percentage or outstanding chips
use rand::{seq::SliceRandom, thread_rng, Rng};

pub fn simulate_icm_tournament(chip_stacks: &[i32], payments: &[i32]) -> Vec<i32> {
    // We're going to mutate in place so move the chip stacks into a mutable vector.
    let mut remaining_stacks: Vec<i32> = chip_stacks.into();
    // Thread local rng.
    let mut rng = thread_rng();
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
            // TODO: Check to see if adding in a skill(running avg of win %) array for each player is needed.
            let hero_won: bool = rng.gen_bool(0.5);

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
        let mut rng = thread_rng();

        for num_players in [2, 3, 4, 5, 15, 16, 32].iter() {
            let chips: Vec<i32> = (0..*num_players).map(|_pn| rng.gen_range(1..500)).collect();

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
            .map(|v| (*v as f64) / (num_trials as f64))
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
                .collect()
        }

        let final_share: Vec<f64> = total_winnings
            .iter()
            .map(|v| (*v as f64) / (num_trials as f64))
            .collect();

        let sum: f64 = final_share.iter().sum();
        let avg = sum / (final_share.len() as f64);

        for &share in final_share.iter() {
            assert!(share < 1.1 * avg);
            assert!(1.1 * share > avg);
        }
    }
}
