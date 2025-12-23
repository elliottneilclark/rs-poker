extern crate rs_poker;

use clap::Parser;
use rs_poker::arena::{
    AgentGenerator, CloneHistorianGenerator, HistorianGenerator,
    agent::{CloneAgentGenerator, RandomAgentGenerator, RandomPotControlAgent},
    competition::{HoldemCompetition, StandardSimulationIterator},
    game_state::RandomGameStateGenerator,
    historian::DirectoryHistorian,
};

#[derive(Parser, Debug)]
#[command(
    name = "agent_battle",
    about = "Run a poker agent battle simulation",
    long_about = "Simulate poker games with various agents competing against each other.\n\
                  Track statistics and save game history to disk."
)]
struct Args {
    /// Number of agents to compete
    #[arg(short = 'a', long, default_value_t = 4)]
    num_agents: usize,

    /// Number of rounds to simulate per batch
    #[arg(short = 'b', long, default_value_t = 500)]
    rounds_per_batch: usize,

    /// Total number of batches to run
    #[arg(short = 'n', long, default_value_t = 5000)]
    num_batches: usize,

    /// Directory to save game history
    #[arg(short = 'd', long, default_value = "historian_out")]
    output_dir: String,

    /// Minimum starting stack (in big blinds)
    #[arg(long, default_value_t = 10.0)]
    min_stack_bb: f32,

    /// Maximum starting stack (in big blinds)
    #[arg(long, default_value_t = 1000.0)]
    max_stack_bb: f32,

    /// Big blind amount
    #[arg(long, default_value_t = 10.0)]
    big_blind: f32,

    /// Small blind amount
    #[arg(long, default_value_t = 5.0)]
    small_blind: f32,
}

fn main() {
    let args = Args::parse();

    println!("Agent Battle Configuration:");
    println!("===========================");
    println!("Number of agents: {}", args.num_agents);
    println!("Rounds per batch: {}", args.rounds_per_batch);
    println!("Total batches: {}", args.num_batches);
    println!("Output directory: {}", args.output_dir);
    println!(
        "Stack range: {}-{} BB",
        args.min_stack_bb, args.max_stack_bb
    );
    println!("Blinds: {}/{}", args.small_blind, args.big_blind);
    println!();

    // Generate agents based on the requested number
    let mut agent_gens: Vec<Box<dyn AgentGenerator>> = Vec::new();
    for i in 0..args.num_agents {
        if i % 2 == 0 {
            agent_gens.push(Box::<RandomAgentGenerator>::default());
        } else {
            // Vary the pot control parameters for different agents
            let params = vec![0.5 - (i as f64 * 0.05), 0.3];
            agent_gens.push(Box::new(CloneAgentGenerator::new(
                RandomPotControlAgent::new(params),
            )));
        }
    }

    // Show how to use the historian to record the games.
    let path = std::env::current_dir().unwrap();
    let dir = path.join(&args.output_dir);
    let hist_gens: Vec<Box<dyn HistorianGenerator>> = vec![Box::new(CloneHistorianGenerator::new(
        DirectoryHistorian::new(dir),
    ))];

    // Convert BB to chip stacks
    let min_stack = args.min_stack_bb * args.big_blind;
    let max_stack = args.max_stack_bb * args.big_blind;

    // Run the games with completely random hands.
    let game_state_gen = RandomGameStateGenerator::new(
        agent_gens.len(),
        min_stack,
        max_stack,
        args.big_blind,
        args.small_blind,
        0.0,
    );
    let simulation_gen = StandardSimulationIterator::new(agent_gens, hist_gens, game_state_gen);
    let mut comp = HoldemCompetition::new(simulation_gen);

    println!("Starting simulation...");
    println!();

    let progress_interval = args.num_batches / 10;
    let progress_interval = if progress_interval < 1 {
        1
    } else {
        progress_interval
    };

    for i in 0..args.num_batches {
        let _res = comp.run(args.rounds_per_batch).expect("competition failed");

        // Show progress stats throughout execution
        if (i + 1) % progress_interval == 0 || i == 0 || i == args.num_batches - 1 {
            let progress_pct = ((i + 1) as f64 / args.num_batches as f64) * 100.0;
            println!(
                "Progress: {}/{} batches ({:.1}%)",
                i + 1,
                args.num_batches,
                progress_pct
            );
            println!("Current Stats: {comp:?}");
            println!();
        }
    }

    println!("Simulation complete!");
    println!("Final Competition Stats: {comp:?}");
}
