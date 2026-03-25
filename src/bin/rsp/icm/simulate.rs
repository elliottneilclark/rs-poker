use clap::Parser;
use rs_poker::core::{RSPokerError};
use rs_poker::simulated_icm::simulate_icm_tournament;
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum SimulateError {
    #[error(transparent)]
    Poker(#[from] RSPokerError),
}

#[derive(Parser, Debug)]
#[command(
    name = "simulate",
    about = "Simulate tournament outcomes to convert chip stacks into $EV",
    long_about = "Run Monte Carlo simulations convert chip EV into $EV.\n\
                  --chips-stacks 100 90 70 60 --payments 5 3 2"
)]
pub struct SimulateArgs {
    /// Chip stacks of all players in the tournament
    #[arg(required = true, long, num_args = 2..)]
    chip_stacks: Vec<f32>,

    /// Prize amounts ($), ordered 1st, 2nd, ...
    #[arg(required = true, long, num_args = 1..)]
    payments: Vec<f32>,

    /// Number of simulations
    #[arg(long, default_value = "100000")]
    iterations: usize
}

pub fn run(args: SimulateArgs) -> Result<(), SimulateError> {
    println!("Starting ICM simulation... {:?} iterations", args.iterations);
    println!("Stacks: {:?}", args.chip_stacks);
    println!("Payments: {:?}", args.payments);
    println!();

    let n = args.chip_stacks.len();

    // Convert the given values, which may include decimals, into centi-units
    let c_chips: Vec<i32> = args.chip_stacks.iter()
        .map(|x| (x * 100.).trunc() as i32)
        .collect();
    
    let c_payments: Vec<i32> = args.payments.iter()
        .map(|x| (x * 100.).trunc() as i32)
        .collect();

    // Run the simulation the designated number of times,
    // then fold each iteration into the final_sums vector.
    let final_sums = (0..args.iterations)
    .into_par_iter()
        .fold(
            || vec![0.0f64; n], // Initializer for each thread's local sum
            |mut local_sums, _| {
                let res = simulate_icm_tournament(c_chips.as_slice(), c_payments.as_slice());
                for i in 0..n {
                    local_sums[i] += res[i] as f64;
                }
                local_sums
            },
        )
        .reduce(
            || vec![0.0f64; n], // Neutral value for combining results
            |mut acc, local| {
                for i in 0..n { acc[i] += local[i]; }
                acc
            },
        );
    
    // Average via the number of iterations, then convert results back to their original scale
    let results: Vec<_> = final_sums.into_iter()
        .map(|c| (0.01 * c as f32) / args.iterations as f32)
        .collect();
        
    println!("Results:");
    println!("========");
    println!("{:?}", results);
    Ok(())
}