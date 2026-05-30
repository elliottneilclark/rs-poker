use std::sync::Arc;

use clap::Parser;
use rs_poker::core::RSPokerError;
use rs_poker::simulated_icm::simulate_icm_tournament;
use tokio::task::JoinSet;

#[derive(Debug, thiserror::Error)]
pub enum SimulateError {
    #[error(transparent)]
    Poker(#[from] RSPokerError),
    #[error("simulation task failed to join: {0}")]
    Join(#[from] tokio::task::JoinError),
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
    iterations: usize,
}

/// Run a chunk of `count` ICM tournament simulations, summing the per-player
/// results into a fresh vector of length `n`.
///
/// Pure CPU work; chunked so each spawned task does meaningful work rather than
/// one task per iteration.
fn simulate_chunk(count: usize, c_chips: &[i32], c_payments: &[i32], n: usize) -> Vec<f64> {
    let mut local_sums = vec![0.0f64; n];
    for _ in 0..count {
        let res = simulate_icm_tournament(c_chips, c_payments);
        for i in 0..n {
            local_sums[i] += res[i] as f64;
        }
    }
    local_sums
}

pub async fn run(args: SimulateArgs) -> Result<(), SimulateError> {
    println!(
        "Starting ICM simulation... {:?} iterations",
        args.iterations
    );
    println!("Stacks: {:?}", args.chip_stacks);
    println!("Payments: {:?}", args.payments);
    println!();

    let n = args.chip_stacks.len();

    // Convert the given values, which may include decimals, into centi-units
    let c_chips: Arc<Vec<i32>> = Arc::new(
        args.chip_stacks
            .iter()
            .map(|x| (x * 100.).trunc() as i32)
            .collect(),
    );

    let c_payments: Arc<Vec<i32>> = Arc::new(
        args.payments
            .iter()
            .map(|x| (x * 100.).trunc() as i32)
            .collect(),
    );

    // Bound concurrency with the CFR engine's default in-flight limiter (sized
    // at `8 × available parallelism`). Each spawned task holds a permit for its
    // lifetime; work is split into chunks so the per-iteration cost dwarfs the
    // spawn/join overhead.
    let semaphore = rs_poker::arena::cfr::build_default_limiter();
    let max_in_flight = rs_poker::arena::cfr::default_limiter_permits();

    // Split the work into roughly `max_in_flight` chunks so every worker gets a
    // turn while keeping the number of spawned tasks small.
    let num_chunks = max_in_flight.min(args.iterations.max(1));
    let base = args.iterations / num_chunks;
    let remainder = args.iterations % num_chunks;

    let mut set: JoinSet<Vec<f64>> = JoinSet::new();
    for chunk_idx in 0..num_chunks {
        // Distribute the remainder across the first `remainder` chunks.
        let count = base + usize::from(chunk_idx < remainder);
        if count == 0 {
            continue;
        }
        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .expect("semaphore never closed");
        let c_chips = Arc::clone(&c_chips);
        let c_payments = Arc::clone(&c_payments);
        set.spawn(async move {
            let _permit = permit;
            // Pure CPU work; run it on a blocking-friendly path via the worker
            // thread the task is scheduled on (each task is short-lived and
            // CPU-bound, gated by the semaphore).
            simulate_chunk(count, &c_chips, &c_payments, n)
        });
    }

    // Reduce the per-task partial sums into the final totals.
    let mut final_sums = vec![0.0f64; n];
    while let Some(joined) = set.join_next().await {
        let local = joined?;
        for i in 0..n {
            final_sums[i] += local[i];
        }
    }

    // Average via the number of iterations, then convert results back to their original scale
    let results: Vec<_> = final_sums
        .into_iter()
        .map(|c| (0.01 * c / args.iterations as f64) as f32)
        .collect();

    println!("Results:");
    println!("========");
    println!("{:?}", results);
    Ok(())
}
